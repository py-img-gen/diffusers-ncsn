import os
import pathlib
from typing import Dict

import pytest
import torch
from diffusers.utils import make_image_grid
from huggingface_hub import HfApi

from ncsn.pipeline_ncsn import NCSNPipeline


@pytest.fixture
def num_inference_steps() -> int:
    return 10


@pytest.fixture
def num_grid_rows() -> int:
    return 4


@pytest.fixture
def num_grid_cols() -> int:
    return 4


@pytest.fixture
def batch_size(num_grid_rows: int, num_grid_cols: int) -> int:
    return num_grid_rows * num_grid_cols


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPUs available for testing"
)
def test_pretrained_pipeline(
    project_dir: pathlib.Path,
    device: torch.device,
    batch_size: int,
    num_inference_steps: int,
    seed: int,
    num_grid_rows: int,
    num_grid_cols: int,
):
    load_path = project_dir / "ncsn-pipeline"

    pipe = NCSNPipeline.from_pretrained(load_path)
    pipe = pipe.to(device)

    def decode_samples(
        pipe: NCSNPipeline,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict,
    ) -> Dict:
        if step_index != 0:
            return callback_kwargs

        # Decode the samples to images
        samples = callback_kwargs["samples"]
        samples = samples.detach().clone()

        samples = pipe.decode_samples(samples)
        images = pipe.numpy_to_pil(samples.cpu().numpy())

        # Create the grid image
        image = make_image_grid(
            images,
            rows=num_grid_rows,
            cols=num_grid_cols,
        )

        # Prepare the directory to save the images
        save_steps_dir = project_dir / "timesteps"
        save_steps_dir.mkdir(parents=True, exist_ok=True)

        # Save the images
        image.save(
            save_steps_dir / f"timestep={timestep:03d}_annealing={step_index:03d}.png"
        )

        return callback_kwargs

    output = pipe(
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        generator=torch.manual_seed(seed),
        callback_on_step_end=decode_samples,  # type: ignore
        callback_on_step_end_tensor_inputs=["samples"],
    )
    image = make_image_grid(
        images=output.images, rows=num_grid_rows, cols=num_grid_cols
    )
    image.save("final.png")

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from PIL import Image

    timesteps_dir = project_dir / "timesteps"
    step_image_paths = list(timesteps_dir.glob("*.png"))

    fig, ax = plt.subplots(figsize=(batch_size, batch_size))
    ax.set_axis_off()

    step_images = [
        [ax.imshow(Image.open(p), animated=True, cmap="gray")] for p in step_image_paths
    ]

    # Repeat the last image for 5 frames to make the gif pause
    step_images += [step_images[-1]] * 5

    # Create and save the animation
    ani = animation.ArtistAnimation(
        fig=fig,
        artists=step_images,
        interval=300,
        repeat_delay=1000,
        blit=True,
        repeat=True,
    )
    ani.save("mnist.gif")


@pytest.fixture
def hf_org_name() -> str:
    return "py-img-gen"


@pytest.fixture
def hf_pipeline_name() -> str:
    return "ncsn-mnist"


@pytest.fixture
def hf_repo_id(hf_org_name: str, hf_pipeline_name: str) -> str:
    return f"{hf_org_name}/{hf_pipeline_name}"


@pytest.fixture
def pipeline_script_path(lib_dir: pathlib.Path) -> pathlib.Path:
    return lib_dir / "pipeline_ncsn.py"


@pytest.fixture
def scheduler_script_path(lib_dir: pathlib.Path) -> pathlib.Path:
    return lib_dir / "scheduling_ncsn.py"


@pytest.fixture
def unet_2d_script_path(lib_dir: pathlib.Path) -> pathlib.Path:
    return lib_dir / "unet_2d_ncsn.py"


def test_push_to_hub(
    hf_repo_id: str,
    pipeline_script_path: pathlib.Path,
    scheduler_script_path: pathlib.Path,
    unet_2d_script_path: pathlib.Path,
    repo_type: str = "model",
) -> None:
    api = HfApi()

    api.upload_file(
        path_or_fileobj=pipeline_script_path,
        path_in_repo=pipeline_script_path.name,
        repo_id=hf_repo_id,
        repo_type=repo_type,
    )
    api.upload_file(
        path_or_fileobj=scheduler_script_path,
        path_in_repo=f"scheduler/{scheduler_script_path.name}",
        repo_id=hf_repo_id,
        repo_type=repo_type,
    )
    api.upload_file(
        path_or_fileobj=unet_2d_script_path,
        path_in_repo=f"unet/{unet_2d_script_path.name}",
        repo_id=hf_repo_id,
        repo_type=repo_type,
    )
