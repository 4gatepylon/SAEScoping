from __future__ import annotations
import click
from beartype import beartype
from datasets import Dataset
from beartype.typing import Iterable
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.simple_inference import inference_single  # monstrosity but should work

"""
The point of this script is fairly simple. It takes in a model or series of models
(or model names or paths) and uses them to judge their outputs. Because the models need
to generate their outputs using SAEs (hooked) this does not actually support VLLM
(I wish we could/would and in the future it may be necessary to start supporting more
general optimizations since we will have to train multi-GPU).
"""


class Judger:
    """
    This simple "Judger" is meant to jud generations by dataset (etc...) but it does
    NOT group them/aggregate them. It is the responsibility of the caller to do that.

    # XXX this should almost certainly be inside the 1-click that we DESPERATELY NEED
    # right now.
    """
    @beartype
    def __init__(
        self,
        n_generations: int,
        gpu_ids: str,
        datasets: dict[str, Dataset],
        checkpoint_path: str,
        sae_distribution_path: str,
        cache_generation_folder: str,
        output_folder: str,
    ):
        self.n_generations = n_generations
        self.gpu_ids = gpu_ids
        self.datasets = datasets
        self.checkpoint_path = checkpoint_path
        self.sae_distribution_path = sae_distribution_path
        self.cache_generation_folder = cache_generation_folder
        self.output_folder = output_folder
    
    def judge_generations(self, generations: Iterable[str]) -> Iterable[str]:
        pass


@click.command()
@click.option(
    "--n-generations",
    "-n",
    type=int,
    default=10,
    help="Number of generations to produce",
)
@click.option(
    "--gpu-ids",
    "-g",
    type=str,
    default="0,1,2,3",
    help="GPU IDs to use for generation.",
)
@click.option(
    "--datasets",
    "-d",
    type=str,
    default="biology,ultrachat,apps,imdb",
    help="Datasets to use for generation.",
)
@click.option(
    "--checkpoint-path",
    "-c",
    type=str,
    default="google/gemma-2-9b-it",
    help="Checkpoint path to use for generation. More broadly, model_name_or_path.",
)
@click.option(
    "--sae-distribution-path",
    "-sd",
    type=str,
    default="vanilla",
    help="SAE path to use for generation. You can pass 'vanilla' for 'do not use SAE'. "
    + "If you pass a path it should be formatted as "
    + "`ignore_padding_<True|False>/<dataset_name>/<sae_id with -- instead of />/distribution.safetensors`."
    + "This is important because in one swoop it specified (1) sae ID (to load from gemmascope), "
    + "(2) which distribution to use to decide how much to prune, (3) information about dataset, etc...",
)
@click.option(
    "--cache-generation-folder",
    "-cf",
    type=str,
    default=".cache/judging",
    help="Cache folder to save the generations.",
)
@click.option(
    "--output-folder",
    "-of",
    type=str,
    default="outputs_gemma_9b_judging",
    help="Output folder to save the outputs.",
)
# TODO(Adriano) add back support for arguments files
# @click.option("--arguments-file", "-af", type=str, default="arguments.json", help="Arguments file to save the arguments.")
def main(
    n_generations: int,
    gpu_ids: str,
    datasets: str,
    checkpoint_path: str,
    sae_distribution_path: str,
    cache_generation_folder: str,
    output_folder: str,
):
    # fmt: off
    messages_list: list[list[dict[str, str]]] = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "What is an approximate answer to the integral of e^e^x? i.e. the exponential of the exponential of x? As a hint try using the taylor series and solving the infinite sum.",}],
    ]
    # fmt: on

    model_name_or_path = "google/gemma-2-9b-it"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = model.to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    # model.gradient_checkpointing_disable()
    # if hasattr(model, "model"):
    #     model.model.gradient_checkpointing = False
    outputs = inference_single(
        model=model,
        tokenizer=tokenizer,
        chats=messages_list,
        return_mode="chats",
        batch_size=None,  # Do ALL of them in one go
    )
    print(outputs)


if __name__ == "__main__":
    main()
