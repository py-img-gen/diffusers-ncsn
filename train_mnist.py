import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.utils import make_image_grid
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from ncsn.pipeline_ncsn import NCSNPipeline
from ncsn.scheduling_ncsn import AnnealedLangevinDynamicScheduler
from ncsn.unet_2d_ncsn import UNet2DModelForNCSN


@dataclass
class CommonConfig(object):
    sigma_min: float = 0.005
    sigma_max: float = 10
    num_train_timesteps: int = 10


@dataclass
class TrainConfig(CommonConfig):
    seed: int = 19950815
    batch_size: int = 256
    num_epochs: int = 200
    display_epoch: int = 10

    num_annealed_steps: int = 100
    sampling_eps: float = 1e-5
    num_workers: int = 4
    shuffle: bool = True
    lr: float = 1e-4

    num_generate_images: int = 16
    num_grid_rows: int = 4
    num_grid_cols: int = 4


@dataclass
class ModelConfig(CommonConfig):
    sample_size: int = 32
    in_channels: int = 1
    out_channels: int = 1
    block_out_channels: Tuple[int, ...] = (64, 128, 256, 512)
    layers_per_block: int = 3
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(sample_size: int) -> transforms.Compose:
    transform_list = [
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def train_iteration(
    train_config: TrainConfig,
    unet: UNet2DModelForNCSN,
    noise_scheduler: AnnealedLangevinDynamicScheduler,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    with tqdm(total=len(data_loader), desc="Iter") as pbar:
        for x, _ in data_loader:
            bsz = x.shape[0]
            x = x.to(device)

            t = torch.randint(
                0,
                train_config.num_train_timesteps,
                size=(bsz,),
                device=device,
            )

            z = torch.randn_like(x)
            x_noisy = noise_scheduler.add_noise(x, z, t)

            scores = unet(x_noisy, t).sample
            used_sigmas = unet.sigmas[t]
            used_sigmas = rearrange(used_sigmas, "b -> b 1 1 1")
            target = -1 / used_sigmas * z

            target = rearrange(target, "b c h w -> b (c h w)")
            scores = rearrange(scores, "b c h w -> b (c h w)")

            loss = F.mse_loss(scores, target, reduction="none")
            loss = loss.mean(dim=-1) * used_sigmas.squeeze() ** 2
            loss = loss.mean(dim=0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update()


def train(
    train_config: TrainConfig,
    unet: UNet2DModelForNCSN,
    noise_scheduler: AnnealedLangevinDynamicScheduler,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    save_dir: Optional[pathlib.Path] = None,
) -> None:
    # Set unet denoiser model to train mode
    unet.train()

    for epoch in tqdm(range(train_config.num_epochs), desc="Epoch"):
        # Run the training iteration
        train_iteration(
            train_config=train_config,
            unet=unet,
            noise_scheduler=noise_scheduler,
            optim=optim,
            data_loader=data_loader,
            device=device,
        )

        # Perform validation and save the model
        if epoch + 1 % train_config.display_epoch == 0 and save_dir is not None:
            # Load the model as a image generation pipeline
            pipe = NCSNPipeline(unet=unet, scheduler=noise_scheduler)
            pipe.set_progress_bar_config(desc="Generating...", leave=False)

            # Generate the images
            output = pipe(
                batch_size=train_config.num_generate_images,
                num_inference_steps=train_config.num_train_timesteps,
                generator=torch.manual_seed(train_config.seed),
            )
            image = make_image_grid(
                images=output.images,  # type: ignore
                rows=train_config.num_grid_rows,
                cols=train_config.num_grid_cols,
            )

            # Save the images
            image.save(save_dir / f"epoch={epoch:03d}.png")
            image.save(save_dir / "training.png")


def main():
    # Create a directory to save the output
    save_dir = pathlib.Path.cwd() / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define the configuration
    train_config = TrainConfig()
    model_config = ModelConfig()

    # Set the seed for reproducibility
    set_seed(seed=train_config.seed)

    # Get the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    unet = UNet2DModelForNCSN(**asdict(model_config))
    unet = unet.to(device)

    # Create the noise scheduler
    noise_scheduler = AnnealedLangevinDynamicScheduler(
        num_train_timesteps=train_config.num_train_timesteps,
        num_annealed_steps=train_config.num_annealed_steps,
        sigma_min=train_config.sigma_min,
        sigma_max=train_config.sigma_max,
        sampling_eps=train_config.sampling_eps,
    )

    # Create the optimizer
    optim = torch.optim.Adam(unet.parameters(), lr=train_config.lr)

    # Load the MNIST dataset
    dataset = torchvision.datasets.MNIST(
        root="~/.cache",
        train=True,
        download=True,
        transform=get_transforms(sample_size=model_config.sample_size),
    )
    # Create the data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
        drop_last=True,
        num_workers=train_config.num_workers,
    )

    # Train the model!
    train(
        train_config=train_config,
        unet=unet,
        noise_scheduler=noise_scheduler,
        optim=optim,
        data_loader=data_loader,
        device=device,
        save_dir=save_dir,
    )

    # Generate the final image
    pipe = NCSNPipeline(unet=unet, scheduler=noise_scheduler)
    output = pipe(
        num_inference_steps=train_config.num_train_timesteps,
        batch_size=train_config.batch_size,
        generator=torch.manual_seed(train_config.seed),
    )

    # Save the final image
    image = make_image_grid(
        images=output.images,  # type: ignore
        rows=train_config.num_grid_rows,
        cols=train_config.num_grid_cols,
    )
    image.save(save_dir / "final.png")


if __name__ == "__main__":
    main()
