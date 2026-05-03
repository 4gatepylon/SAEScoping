"""Sweep scheduler: compiles the dependency graph and dispatches step jobs.

Inputs:
    --experiment-config: path to an ExperimentConfig YAML (e.g. mini_test.yaml)
    --devices: comma-separated CUDA devices (e.g. "cuda:0,cuda:1")
    Environment: $SAESCOPING_ARTIFACTS_LOCATION (base artifacts dir)

Outputs (under $SAESCOPING_ARTIFACTS_LOCATION/{artifacts_subdir}/):
    runtime_state_mirror/dependency_graph.yaml — serialized DAG (written once)
    runtime_state_mirror/operations.jsonl — append-only state transitions
    runtime_state_mirror/state_v{NNN}_{ISO}.yaml — periodic full-state snapshots

Side effects:
    Launches calibrate.py / pgd_or_elicit.py subprocesses on assigned GPUs.
    Logs high-level progress to stdout.

Idempotency:
    Steps whose output artifacts already exist are marked completed and skipped.
    Re-running the scheduler continues from where it left off.

Failure mode:
    If a step subprocess exits non-zero, it is marked failed. All dependents
    are transitively skipped. The scheduler continues with independent jobs.

TODO(hadriano) not reviewed so might just not work tbh
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import click
import yaml

from interface import (
    CalibrateStep,
    DependencyGraph,
    DependencyGraphNode,
    ElicitStep,
    ExperimentConfig,
    ModelConfig,
    PGDStep,
    Step,
    StepSpec,
    _slash_safe,
    make_step_id,
    wandb_run_name,
)

# ── Constants ─────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_BYTES_PER_PARAM_CHECKPOINT = 4  # fp32 optimizer states dominate


# ── State management ──────────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _is_complete_checkpoint(ckpt_path: Path) -> bool:
    """Check that a HuggingFace checkpoint dir has all model weights + tokenizer.

    Handles both single-file (model.safetensors) and sharded
    (model.safetensors.index.json + all listed shards) formats.
    A partial save (e.g. interrupted mid-write) will be missing the
    index file or some shard files.
    """
    files = {f.name for f in ckpt_path.iterdir()}
    has_tokenizer = "tokenizer.json" in files or "tokenizer.model" in files
    if not has_tokenizer:
        return False
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return True
    index_file = ckpt_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text())
            needed_shards = set(index.get("weight_map", {}).values())
            return all(shard in files for shard in needed_shards)
        except (json.JSONDecodeError, OSError):
            return False
    return False


class SchedulerState:
    """Tracks job statuses and GPU availability."""

    def __init__(
        self,
        graph: DependencyGraph,
        devices: list[str],
        artifacts_root: Path,
        experiment_config: ExperimentConfig,
        model_configs: dict[str, ModelConfig],
        no_cache: bool,
        dry_run: bool = False,
    ):
        self.graph = graph
        self.devices = devices
        self.artifacts_root = artifacts_root
        self.experiment_config = experiment_config
        self.model_configs = model_configs
        self.no_cache = no_cache
        self.dry_run = dry_run

        self._status: dict[str, JobStatus] = {}
        self._node_map: dict[str, DependencyGraphNode] = {}
        for node in graph.nodes:
            self._node_map[node.step_id] = node
            self._status[node.step_id] = JobStatus.PENDING

        self._gpu_in_use: dict[str, Optional[str]] = {d: None for d in devices}
        self._running_procs: dict[str, subprocess.Popen] = {}
        self._running_device: dict[str, str] = {}
        self._running_logs: dict[str, Any] = {}

        self._mirror_dir = artifacts_root / "runtime_state_mirror"
        self._mirror_dir.mkdir(parents=True, exist_ok=True)
        self._ops_file = self._mirror_dir / "operations.jsonl"
        self._snapshot_idx = 0
        self._completions_since_snapshot = 0
        self._snapshot_every_n_completions = 4

    # ── Cache detection ───────────────────────────────────────────────────

    def _step_output_exists(self, step: Step) -> bool:
        """Check if a step's primary output already exists on disk.

        Primary signal: COMPLETED marker written as the very last action by
        each callee. Sanity check for PGD: also verify at least one checkpoint
        has valid weights.
        """
        if isinstance(step, CalibrateStep):
            output_dir = self.artifacts_root / "saliency_maps" / _slash_safe(step.model_id) / step.scope_domain
            return (output_dir / "COMPLETED").exists()
        elif isinstance(step, PGDStep):
            ckpt_dir = self.artifacts_root / step.checkpoint_dir
            if not (ckpt_dir / "COMPLETED").exists():
                return False
            for sub in ckpt_dir.iterdir():
                if sub.is_dir() and sub.name.startswith("checkpoint-"):
                    if _is_complete_checkpoint(sub):
                        return True
            return False
        elif isinstance(step, ElicitStep):
            model_safe = _slash_safe(step.model_id)
            logs_dir = self.artifacts_root / "elicitation_judge_logs" / model_safe / step.scope_domain / step.elicitation_domain / str(step.sparsity)
            return (logs_dir / "COMPLETED").exists()
        return False

    def mark_cached_steps(self) -> None:
        """Pre-scan: mark steps whose outputs exist as completed."""
        if self.no_cache:
            return
        n_cached = 0
        for step_id, node in self._node_map.items():
            if self._step_output_exists(node.step):
                self._status[step_id] = JobStatus.COMPLETED
                self._log_operation(step_id, "completed", "cached output exists")
                print(f"[scheduler] SKIP  {wandb_run_name(node.step)} (cached)")
                n_cached += 1
        if n_cached:
            print(f"[scheduler] Skipped {n_cached} steps with existing outputs")

    # ── Scheduling logic ──────────────────────────────────────────────────

    def _deps_satisfied(self, step_id: str) -> bool:
        node = self._node_map[step_id]
        for dep_id in node.deps:
            if self._status[dep_id] != JobStatus.COMPLETED:
                return False
        return True

    def _deps_have_failure(self, step_id: str) -> bool:
        node = self._node_map[step_id]
        for dep_id in node.deps:
            if self._status[dep_id] in (JobStatus.FAILED, JobStatus.SKIPPED):
                return True
        return False

    def _get_free_device(self) -> Optional[str]:
        for dev, occupant in self._gpu_in_use.items():
            if occupant is None:
                return dev
        return None

    def _get_ready_jobs(self) -> list[str]:
        ready = []
        for step_id, status in self._status.items():
            if status != JobStatus.PENDING:
                continue
            if self._deps_have_failure(step_id):
                self._status[step_id] = JobStatus.SKIPPED
                self._log_operation(step_id, "skipped", "upstream failure")
                node = self._node_map[step_id]
                print(f"[scheduler] SKIP  {wandb_run_name(node.step)} (upstream failure)")
                continue
            if self._deps_satisfied(step_id):
                ready.append(step_id)
        return ready

    # TODO(hadriano) review this for bugs
    def _compile_step_spec(self, step: Step, device: str) -> StepSpec:
        """Build a fully-resolved StepSpec for a step (all config pre-merged)."""
        mc = self.model_configs[step.model_id]
        merged_sft = {**mc.sft, **self.experiment_config.sft_overrides}
        merged_mc = mc.model_copy(update={"sft": merged_sft})
        return StepSpec(
            step=step,
            model_cfg=merged_mc,
            dataset_name=self.experiment_config.dataset_name,
            scope_domains=self.experiment_config.scope_domains,
            n_calibration=self.experiment_config.n_calibration,
            n_train=self.experiment_config.n_train,
            n_eval=self.experiment_config.n_eval,
            calibration_sweep_sparsities=self.experiment_config.calibration_sweep_sparsities,
            artifacts_subdir=self.experiment_config.operational.artifacts_subdir,
            no_cache=self.experiment_config.operational.no_cache,
            save_elicitation_checkpoints=self.experiment_config.operational.save_elicitation_checkpoints,
            wandb=self.experiment_config.operational.wandb,
            llm_judge=self.experiment_config.operational.llm_judge,
            device=device,
            dry_run=self.dry_run,
        )

    def _write_step_spec(self, step_id: str, spec: StepSpec) -> Path:
        """Serialize a StepSpec to the state mirror and return its path."""
        specs_dir = self._mirror_dir / "step_specs"
        specs_dir.mkdir(parents=True, exist_ok=True)
        spec_path = specs_dir / f"{step_id}.yaml"
        with open(spec_path, "w") as f:
            yaml.safe_dump(
                json.loads(spec.model_dump_json()),
                f,
                default_flow_style=False,
            )
        return spec_path

    def _launch_step(self, step_id: str, device: str) -> None:
        node = self._node_map[step_id]
        step = node.step
        spec = self._compile_step_spec(step, "cuda:0")
        spec_path = self._write_step_spec(step_id, spec)
        cmd = self._build_command(step, spec_path)
        tag = "DRY-RUN " if self.dry_run else ""
        print(f"[scheduler] {tag}LAUNCH {wandb_run_name(step)} on {device}")
        print(f"            cmd: {' '.join(cmd)}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device.replace("cuda:", "")
        log_path = self._mirror_dir / "logs" / f"{step_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        self._status[step_id] = JobStatus.RUNNING
        self._gpu_in_use[device] = step_id
        self._running_procs[step_id] = proc
        self._running_device[step_id] = device
        self._running_logs[step_id] = log_fh
        self._log_operation(step_id, "started", f"device={device}, pid={proc.pid}, log={log_path}")

    def _build_command(self, step: Step, spec_path: Path) -> list[str]:
        if isinstance(step, CalibrateStep):
            script = "calibrate.py"
        else:
            script = "pgd_or_elicit.py"
        return [sys.executable, str(_HERE / script), "--step-spec", str(spec_path)]

    def _poll_running(self) -> None:
        """Check running processes; reap any that finished."""
        done = []
        for step_id, proc in self._running_procs.items():
            rc = proc.poll()
            if rc is not None:
                done.append((step_id, rc, proc))
        for step_id, rc, proc in done:
            device = self._running_device[step_id]
            self._gpu_in_use[device] = None
            log_fh = self._running_logs.pop(step_id, None)
            if log_fh:
                log_fh.close()
            del self._running_procs[step_id]
            del self._running_device[step_id]
            log_path = self._mirror_dir / "logs" / f"{step_id}.log"
            node = self._node_map[step_id]
            if rc == 0:
                self._status[step_id] = JobStatus.COMPLETED
                self._log_operation(step_id, "completed", f"exit_code=0")
                print(f"[scheduler] DONE  {wandb_run_name(node.step)}")
            else:
                self._status[step_id] = JobStatus.FAILED
                self._log_operation(step_id, "failed", f"exit_code={rc}")
                tail = ""
                try:
                    tail = log_path.read_text()[-2000:]
                except OSError:
                    tail = "(could not read log)"
                print(f"[scheduler] FAIL  {wandb_run_name(node.step)} (rc={rc})")
                print(f"            log: {log_path}")
                print(f"            last output:\n{tail}")
            self._completions_since_snapshot += 1
            if self._completions_since_snapshot >= self._snapshot_every_n_completions:
                self._save_snapshot()
                self._completions_since_snapshot = 0

    def run(self) -> dict[str, JobStatus]:
        """Main dispatch loop. Returns final status map."""
        self.mark_cached_steps()
        self._save_snapshot()

        while True:
            self._poll_running()

            ready = self._get_ready_jobs()
            for step_id in ready:
                device = self._get_free_device()
                if device is None:
                    break
                self._launch_step(step_id, device)

            all_terminal = all(s in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.SKIPPED) for s in self._status.values())
            if all_terminal and not self._running_procs:
                break

            time.sleep(2.0)

        self._save_snapshot()
        return dict(self._status)

    # ── Logging / state mirror ────────────────────────────────────────────

    def _log_operation(self, step_id: str, event: str, detail: str = "") -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step_id": step_id,
            "event": event,
            "detail": detail,
        }
        with open(self._ops_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _save_snapshot(self) -> None:
        self._snapshot_idx += 1
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"state_v{self._snapshot_idx:03d}_{ts}.yaml"
        snapshot = {
            "statuses": {sid: s.value for sid, s in self._status.items()},
            "gpu_assignments": {d: sid for d, sid in self._gpu_in_use.items() if sid},
        }
        with open(self._mirror_dir / filename, "w") as f:
            yaml.safe_dump(snapshot, f, default_flow_style=False)


# ── Graph compilation ─────────────────────────────────────────────────────


def _compile_graph(
    experiment: ExperimentConfig,
    model_configs: dict[str, ModelConfig],
) -> DependencyGraph:
    """Build the full dependency DAG from the experiment config."""
    nodes: list[DependencyGraphNode] = []
    calibrate_ids: dict[tuple[str, str], str] = {}
    pgd_ids: dict[tuple[str, str, float], str] = {}

    scope_domains = experiment.scope_domains
    sparsities = experiment.sparsities

    if experiment.elicitation_domains is not None:
        elicitation_domains_for = {sd: experiment.elicitation_domains for sd in scope_domains}
    else:
        elicitation_domains_for = {sd: [d for d in scope_domains if d != sd] for sd in scope_domains}

    for model_id in [mc.model_id for mc in model_configs.values()]:
        model_safe = _slash_safe(model_id)

        for scope_domain in scope_domains:
            saliency_rel = f"saliency_maps/{model_safe}/{scope_domain}/wanda_saliency.safetensors"

            cal_step = CalibrateStep(
                model_id=model_id,
                scope_domain=scope_domain,
                saliency_path=saliency_rel,
            )
            cal_id = make_step_id(cal_step)
            calibrate_ids[(model_id, scope_domain)] = cal_id
            nodes.append(DependencyGraphNode(step_id=cal_id, step=cal_step, deps=[]))

            for sparsity in sparsities:
                ckpt_rel = f"pgd_checkpoints/{model_safe}/{scope_domain}/{sparsity}"

                pgd_step = PGDStep(
                    model_id=model_id,
                    scope_domain=scope_domain,
                    sparsity=sparsity,
                    saliency_path=saliency_rel,
                    checkpoint_dir=ckpt_rel,
                )
                pgd_id = make_step_id(pgd_step)
                pgd_ids[(model_id, scope_domain, sparsity)] = pgd_id
                nodes.append(DependencyGraphNode(step_id=pgd_id, step=pgd_step, deps=[cal_id]))

                for elicit_domain in elicitation_domains_for[scope_domain]:
                    elicit_ckpt_rel = f"elicitation_checkpoints/{model_safe}/{scope_domain}/{elicit_domain}/{sparsity}"
                    elicit_step = ElicitStep(
                        model_id=model_id,
                        scope_domain=scope_domain,
                        sparsity=sparsity,
                        elicitation_domain=elicit_domain,
                        pgd_checkpoint_dir=ckpt_rel,
                        checkpoint_dir=elicit_ckpt_rel,
                    )
                    elicit_id = make_step_id(elicit_step)
                    nodes.append(DependencyGraphNode(step_id=elicit_id, step=elicit_step, deps=[pgd_id]))

    return DependencyGraph(experiment_name=experiment.name, nodes=nodes)


# ── Graph pretty-printing ─────────────────────────────────────────────────


def _step_label(step: Step) -> str:
    if isinstance(step, CalibrateStep):
        return f"calibrate {step.scope_domain}"
    elif isinstance(step, PGDStep):
        return f"pgd {step.scope_domain} s={step.sparsity}"
    elif isinstance(step, ElicitStep):
        return f"elicit {step.scope_domain}→{step.elicitation_domain} s={step.sparsity}"
    return str(step)


def _print_graph_tree(graph: DependencyGraph) -> None:
    """Print the dependency graph as a unicode tree to stdout."""
    id_to_node = {n.step_id: n for n in graph.nodes}
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for node in graph.nodes:
        if not node.deps:
            roots.append(node.step_id)
        for dep in node.deps:
            children[dep].append(node.step_id)

    # Group roots by model_id for readability
    model_roots: dict[str, list[str]] = defaultdict(list)
    for rid in roots:
        model_roots[id_to_node[rid].step.model_id].append(rid)

    def _walk(node_id: str, prefix: str, is_last: bool) -> None:
        node = id_to_node[node_id]
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{_step_label(node.step)}  [{node.step_id[:8]}]")
        child_prefix = prefix + ("    " if is_last else "│   ")
        kids = children.get(node_id, [])
        for i, kid in enumerate(kids):
            _walk(kid, child_prefix, i == len(kids) - 1)

    for model_id, rids in model_roots.items():
        print(f"\n{model_id}  ({len(graph.nodes)} nodes total)")
        for i, rid in enumerate(rids):
            _walk(rid, "", i == len(rids) - 1)


# ── Pre-flight checks ─────────────────────────────────────────────────────


def _estimate_disk_usage(
    graph: DependencyGraph,
    model_configs: dict[str, ModelConfig],
    save_elicitation: bool,
) -> int:
    """Rough estimate of total bytes that will be written."""
    total = 0
    for node in graph.nodes:
        step = node.step
        if isinstance(step, CalibrateStep):
            total += 200_000_000  # ~200MB saliency map
        elif isinstance(step, PGDStep):
            mc = model_configs.get(step.model_id)
            if mc:
                param_count = _estimate_param_count(mc.model_id)
                assert mc.sft.get("max_steps") is not None and mc.sft.get("save_steps") is not None
                n_checkpoints = max(1, mc.sft["max_steps"] // mc.sft["save_steps"])
                total += n_checkpoints * param_count * _BYTES_PER_PARAM_CHECKPOINT
        elif isinstance(step, ElicitStep) and save_elicitation:
            mc = model_configs.get(step.model_id)
            if mc:
                param_count = _estimate_param_count(mc.model_id)
                total += param_count * _BYTES_PER_PARAM_CHECKPOINT
    return total


def _estimate_param_count(model_id: str) -> int:
    """Rough parameter counts for known models."""
    model_id_lower = model_id.lower()
    if "2b" in model_id_lower:
        return 2_000_000_000
    elif "4b" in model_id_lower:
        return 4_000_000_000
    elif "9b" in model_id_lower:
        return 9_000_000_000
    elif "12b" in model_id_lower:
        return 12_000_000_000
    return 7_000_000_000  # default guess


def _preflight_disk_check(artifacts_root: Path, estimated_bytes: int) -> None:
    """Abort if estimated usage exceeds 90% of free space."""
    usage = shutil.disk_usage(artifacts_root)
    if estimated_bytes > usage.free * 0.9:
        required_gb = estimated_bytes / (1024**3)
        free_gb = usage.free / (1024**3)
        raise click.ClickException(
            f"Disk space check failed: estimated that it woudl need {required_gb:.1f} GB, "
            f"but only {free_gb:.1f} GB free (90% threshold). "
            f"Free up space or reduce the sweep."
        )
    free_gb = usage.free / (1024**3)
    est_gb = estimated_bytes / (1024**3)
    print(f"[scheduler] Disk check OK: ~{est_gb:.1f} GB estimated, {free_gb:.1f} GB free")


# ── CLI entry point ───────────────────────────────────────────────────────


def _resolve_artifacts_root(experiment: ExperimentConfig) -> Path:
    base = os.environ.get("SAESCOPING_ARTIFACTS_LOCATION")
    if not base:
        raise click.ClickException("SAESCOPING_ARTIFACTS_LOCATION not set. Export it to point to your artifacts directory.")
    root = Path(base) / experiment.operational.artifacts_subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_model_configs(
    experiment: ExperimentConfig,
    experiment_yaml_dir: Path,
) -> dict[str, ModelConfig]:
    """Load all model config YAMLs referenced by the experiment."""
    configs: dict[str, ModelConfig] = {}
    for fname in experiment.model_configs:
        path = experiment_yaml_dir / fname
        if not path.exists():
            raise click.ClickException(f"Model config not found: {path}")
        mc = ModelConfig.from_yaml(path)
        configs[mc.model_id] = mc
    return configs


@click.command()
@click.option(
    "--experiment-config",
    required=True,
    type=click.Path(exists=True),
    help="Path to the experiment YAML (e.g. mini_test.yaml)",
)
@click.option(
    "--devices",
    default="cuda:0",
    help="Comma-separated CUDA devices for the GPU pool",
)
@click.option(
    "--exit-early",
    type=click.Choice(["graph", "disk", "end"], case_sensitive=False),
    default="end",
    help="Exit after a phase: 'graph' (after writing DAG), 'disk' (after disk check), 'end' (full run)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Load models to CPU and write outputs without training (tests disk layout + scheduler dispatch)",
)
def main(experiment_config: str, devices: str, exit_early: str, dry_run: bool) -> None:
    """Compile dependency graph and dispatch sweep jobs across GPUs."""
    exp_path = Path(experiment_config).resolve()
    experiment = ExperimentConfig.from_yaml(exp_path)
    device_list = [d.strip() for d in devices.split(",")]

    model_configs = _load_model_configs(experiment, exp_path.parent)
    artifacts_root = _resolve_artifacts_root(experiment)

    if dry_run:
        print("[scheduler] *** DRY RUN — models loaded to CPU, no training, outputs are stubs ***")
    print(f"[scheduler] Experiment: {experiment.name}")
    print(f"[scheduler] Models: {list(model_configs.keys())}")
    print(f"[scheduler] Devices: {device_list}")
    print(f"[scheduler] Artifacts: {artifacts_root}")

    graph = _compile_graph(experiment, model_configs)
    print(f"[scheduler] Compiled graph: {len(graph.nodes)} nodes")

    # Write dependency graph (once, never modified)
    mirror_dir = artifacts_root / "runtime_state_mirror"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    graph_path = mirror_dir / "dependency_graph.yaml"
    if not graph_path.exists():
        with open(graph_path, "w") as f:
            yaml.safe_dump(
                json.loads(graph.model_dump_json()),
                f,
                default_flow_style=False,
            )
        print(f"[scheduler] Wrote {graph_path}")

    if exit_early == "graph":
        _print_graph_tree(graph)
        print("\n[scheduler] --exit-early=graph: stopping after graph compile.")
        return

    # Pre-flight disk check
    estimated = _estimate_disk_usage(graph, model_configs, experiment.operational.save_elicitation_checkpoints)
    _preflight_disk_check(artifacts_root, estimated)

    if exit_early == "disk":
        print("[scheduler] --exit-early=disk: stopping after disk check.")
        return

    # Run
    state = SchedulerState(
        graph=graph,
        devices=device_list,
        artifacts_root=artifacts_root,
        experiment_config=experiment,
        model_configs=model_configs,
        no_cache=experiment.operational.no_cache,
        dry_run=dry_run,
    )
    final = state.run()

    # Summary
    counts = defaultdict(int)
    for s in final.values():
        counts[s] += 1
    print(f"\n[scheduler] DONE. Summary:")
    for status, count in sorted(counts.items(), key=lambda x: x[0].value):
        print(f"  {status.value}: {count}")

    n_failed = counts.get(JobStatus.FAILED, 0)
    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
