from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import click


DEFAULT_SOURCE = Path("/mnt/align4_drive/arunas/sae-filters/SAEScoping/wandb")
DEFAULT_IDS_FILE = Path(__file__).resolve().parent / "aruna-evals-fresh.json"
DEFAULT_DEST_SUBDIR = "aruna_wandb"

_ID_RE = re.compile(r"\b[a-z0-9]{8}\b")

# Regression baseline: every ID listed by the user on 2026-05-07.
# The parser must keep returning a superset of this list. New IDs may be
# added freely; the assert in `parse_ids_checked` only fails if one of
# these stops being recognized.
_KNOWN_IDS_2026_05_07: tuple[str, ...] = (
    "va2arbpk", "efduc1pn", "lkv6tn5l", "s7g358gt",
    "12h5ydak", "4n6ch8xe", "bpxxkqkf",
    "uzgczdle", "ifeeknae", "vmi84cw1", "xz4e78xm",
    "01bnrdbp", "ti9xjma3", "lgcpc65b",
    "02jhva3w", "1yzkszvp", "z34n7esx",
    "yoj0m953", "n3ys1j9m", "2oord3it", "xbyexiny",
    "fht7pa1l", "pr328j9w", "jj75i299", "q1umaz8e",
    "2b63aec8",
    "fitoo774",
)


def parse_ids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _ID_RE.finditer(text):
        token = m.group(0)
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def parse_ids_checked(text: str) -> list[str]:
    ids = parse_ids(text)
    parsed_set = set(ids)
    missing = [k for k in _KNOWN_IDS_2026_05_07 if k not in parsed_set]
    assert not missing, (
        f"Parser regression: known IDs not found in input: {missing}. "
        f"Either the IDs file lost them or the parser changed."
    )
    return ids


@click.command()
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_SOURCE,
    show_default=True,
    help="Folder containing the source `run-YYYYMMDD_HHMMSS-<id>` wandb dirs.",
)
@click.option(
    "--ids-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_IDS_FILE,
    show_default=True,
    help="File whose 8-char lowercase alphanumeric tokens are treated as wandb run IDs.",
)
@click.option(
    "--dest",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=f"Override destination. Default: $SAESCOPING_ARTIFACTS_LOCATION/{DEFAULT_DEST_SUBDIR}",
)
@click.option(
    "--no-dry-run",
    "execute",
    is_flag=True,
    default=False,
    help="Actually copy. Default is dry-run (only prints what would happen).",
)
def main(source: Path, ids_file: Path, dest: Path | None, execute: bool) -> None:
    if dest is None:
        loc = os.environ.get("SAESCOPING_ARTIFACTS_LOCATION")
        if loc is None:
            raise click.ClickException(
                "SAESCOPING_ARTIFACTS_LOCATION is not set and --dest was not given."
            )
        dest = Path(loc) / DEFAULT_DEST_SUBDIR

    text = ids_file.read_text()
    ids = parse_ids_checked(text)

    matches: dict[str, list[Path]] = {}
    for i in ids:
        matches[i] = sorted(source.glob(f"run-*-{i}"))

    found = [i for i, h in matches.items() if h]
    missing = [i for i, h in matches.items() if not h]

    click.echo(f"Source:    {source}")
    click.echo(f"IDs file:  {ids_file}")
    click.echo(f"Dest:      {dest}")
    click.echo(f"Parsed {len(ids)} unique IDs; {len(found)} matched, {len(missing)} missing.")
    if missing:
        click.echo(f"Missing IDs (no matching run dir): {missing}")

    if execute:
        click.echo(">>> COPY MODE (--no-dry-run); creating dest if needed.")
        dest.mkdir(parents=True, exist_ok=True)
    else:
        click.echo(">>> DRY-RUN MODE (pass --no-dry-run to actually copy).")

    n_copied = 0
    n_skipped = 0
    for i, hits in matches.items():
        for h in hits:
            target = dest / h.name
            if execute:
                if target.exists():
                    click.echo(f"  skip (exists): {target.name}")
                    n_skipped += 1
                else:
                    shutil.copytree(h, target)
                    click.echo(f"  copied:        {h.name}")
                    n_copied += 1
            else:
                click.echo(f"  would copy:    {h.name}  ->  {target}")

    if execute:
        click.echo(f"Done. copied={n_copied} skipped={n_skipped} missing={len(missing)}")


if __name__ == "__main__":
    main()
