import subprocess
import os
import sys
from enum import Enum
from pathlib import Path
import click


class Mode(str, Enum):
    Full = "full"
    Medium = "medium"
    Light = "light"


@click.command()
@click.option("--profiling-file", type=str, default="experiments/demo.py", help="File for start testing.")
@click.option("--output-name", type=str, default="demo", help="Stem of output files with profiling info.")
@click.option(
    "--mode", type=Mode, default=Mode.Full, 
    help="Mode for profile: 1: Full - with CPU, GPU, CUDA API ...; 2: Medium - GPU inspect with memory control ...; "
         "3: Light - Base measurement of cuda, cublas, etc. using."
)
def profile_with_nsight_systems(profiling_file: str, output_name: str, mode: Mode) -> None:
    root = Path(__file__).parent
    command = f"nsys profile* --output={output_name} {sys.executable} {root}/{profiling_file}"
    flags = {
        "trace": ["cuda", "nvtx", "cublas", "cublas-verbose", "cusparse", "cusparse-verbose"],
        "force-overwrite": "true",
        "stats": "true"
    }

    if mode == Mode.Full:
        flags.update({
            "trace": ["cuda", "nvtx", "cublas", "cublas-verbose", "cusparse", "cusparse-verbose", "opengl",
                      "opengl-annotations", "nvvideo", "vulkan", "vulkan-annotations", "dx11", "dx11-annotations",
                      "dx12", "dx12-annotations", "wddm"],
            "cuda-memory-usage": "true",
            "gpuctxsw": "true"
        })
    elif mode == Mode.Medium:
        flags.update({
            "cuda-memory-usage": "true",
            "gpuctxsw": "true"
        })

    optional_commands = ''
    for flag, value in flags.items():
        if "trace" in flag:
            optional_commands += f" --{flag} {','.join(value)}"
        else:
            optional_commands += f" --{flag}={value}"

    command = command.replace("*", optional_commands)
    click.echo(f"Execution path: {os.getcwd()}")
    click.echo(f"Execution command: {command}")

    subprocess.call(command.split())


if __name__ == "__main__":
    profile_with_nsight_systems()
