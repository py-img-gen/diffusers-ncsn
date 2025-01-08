import pathlib
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision
from diffusers.utils import make_image_grid
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed

from ncsn.pipeline_ncsn import NCSNPipeline
from ncsn.scheduler import AnnealedLangevinDynamicsScheduler
from ncsn.unet import UNet2DModelForNCSN

# Set the dynamic_ncols=True for tqdm
tqdm = partial(tqdm, dynamic_ncols=True)


@dataclass
class TrainArgs(TrainingArguments):
    """Arguments for training the model"""

    output_dir: pathlib.Path = field(
        default=pathlib.Path(__file__).parent / "outputs",
        metadata={"help": "The output directory"},
    )
    per_device_train_batch_size: int = field(
        default=256,
        metadata={"help": "Batch size"},
    )
    num_train_epochs: int = field(
        default=150,
        metadata={"help": "Number of epochs"},
    )
    eval_epoch: int = field(
        default=10,
        metadata={"help": "Epoch to evaluate"},
    )
    num_train_timesteps: int = field(
        default=10,
        metadata={"help": "Number of timesteps"},
    )
    num_annealed_steps: int = field(
        default=100,
        metadata={"help": "Number of annealed steps"},
    )
    sampling_eps: float = field(
        default=1e-5,
        metadata={"help": "Sampling epsilon"},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"},
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle the dataset"},
    )

    def __post_init__(self) -> None:
        self.output_dir = pathlib.Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.animation_dir.mkdir(parents=True, exist_ok=True)

    @property
    def batch_size(self) -> int:
        return self.per_device_train_batch_size

    @property
    def validation_dir(self) -> pathlib.Path:
        return self.output_dir / "validation"

    @property
    def animation_dir(self) -> pathlib.Path:
        return self.output_dir / "timesteps"


@dataclass
class ValidArgs(object):
    """Arguments for validation"""

    num_grid_rows: int = field(
        default=4,
        metadata={"help": "Number of grid rows"},
    )
    num_grid_cols: int = field(
        default=4,
        metadata={"help": "Number of grid columns"},
    )
    num_generate_images: int = field(
        default=16,
        metadata={"help": "Number of images to generate"},
    )


@dataclass
class ModelArgs:
    """Arguments for the model"""

    sigma_min: float = field(
        default=0.005,
        metadata={"help": "Minimum value of sigma"},
    )
    sigma_max: float = field(
        default=10,
        metadata={"help": "Maximum value of sigma"},
    )
    sample_size: int = field(
        default=32,
        metadata={"help": "Size of the input image"},
    )
    in_channels: int = field(
        default=1,
        metadata={"help": "Number of input channels"},
    )
    out_channels: int = field(
        default=1,
        metadata={"help": "Number of output channels"},
    )
    block_out_channels: Tuple[int, ...] = field(
        default=(64, 128, 256, 512),
        metadata={"help": "Number of output channels for each block"},
    )
    layers_per_block: int = field(
        default=3, metadata={"help": "Number of layers per block"}
    )
    down_block_types: Tuple[str, ...] = field(
        default=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        metadata={"help": "Types of down blocks"},
    )
    up_block_types: Tuple[str, ...] = field(
        default=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        metadata={"help": "Types of up blocks"},
    )


def get_transforms(sample_size: int) -> transforms.Compose:
    transform_list = [
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transform_list)


def train_iteration(
    train_args: TrainArgs,
    unet: UNet2DModelForNCSN,
    noise_scheduler: AnnealedLangevinDynamicsScheduler,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    with tqdm(total=len(data_loader), desc="Iteration", leave=False) as pbar:
        for x, _ in data_loader:
            bsz = x.shape[0]
            x = x.to(device)

            # Sample a random timestep
            t = torch.randint(
                0,
                train_args.num_train_timesteps,
                size=(bsz,),
                device=device,
            )

            # Sample a random noise
            z = torch.randn_like(x)
            # Add noise to the input
            x_noisy = noise_scheduler.add_noise(x, z, t)

            # Calculate the score using the model
            scores = unet(x_noisy, t).sample  # type: ignore
            # Calculate the target score

            used_sigmas = unet.sigmas[t]
            used_sigmas = rearrange(used_sigmas, "b -> b 1 1 1")
            target = -1 / used_sigmas * z
            # Rearrange the tensors
            target = rearrange(target, "b c h w -> b (c h w)")
            scores = rearrange(scores, "b c h w -> b (c h w)")

            # Calculate the loss
            loss = F.mse_loss(scores, target, reduction="none")
            loss = loss.mean(dim=-1) * used_sigmas.squeeze() ** 2
            loss = loss.mean(dim=0)

            # Perform the optimization step
            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update()


def train(
    train_args: TrainArgs,
    valid_args: ValidArgs,
    unet: UNet2DModelForNCSN,
    noise_scheduler: AnnealedLangevinDynamicsScheduler,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
) -> None:
    # Set unet denoiser model to train mode
    unet.train()  # type: ignore

    for epoch in tqdm(range(train_args.num_train_epochs), desc="Epoch"):
        # Run the training iteration
        train_iteration(
            train_args=train_args,
            unet=unet,
            noise_scheduler=noise_scheduler,
            optim=optim,
            data_loader=data_loader,
            device=device,
        )

        # Perform validation and save the model
        if epoch % train_args.eval_epoch == 0 and train_args.output_dir is not None:
            # Load the model as a image generation pipeline
            pipe = NCSNPipeline(unet=unet, scheduler=noise_scheduler)
            pipe.set_progress_bar_config(desc="Generating...", leave=False)

            # Generate the images
            output = pipe(
                batch_size=valid_args.num_generate_images,
                num_inference_steps=train_args.num_train_timesteps,
                generator=torch.manual_seed(train_args.seed),
            )
            image = make_image_grid(
                images=output.images,  # type: ignore
                rows=valid_args.num_grid_rows,
                cols=valid_args.num_grid_cols,
            )

            # Save the images
            image.save(train_args.validation_dir / f"epoch={epoch:03d}.png")
            image.save(train_args.validation_dir / "validation.png")


def main(model_args: ModelArgs, train_args: TrainArgs, valid_args: ValidArgs):
    # Set the seed for reproducibility
    set_seed(seed=train_args.seed, deterministic=True)

    # Get the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    unet = UNet2DModelForNCSN(
        num_train_timesteps=train_args.num_train_timesteps, **asdict(model_args)
    )
    unet = unet.to(device)

    # Create the noise scheduler
    noise_scheduler = AnnealedLangevinDynamicsScheduler(
        num_train_timesteps=train_args.num_train_timesteps,
        num_annealed_steps=train_args.num_annealed_steps,
        sigma_min=model_args.sigma_min,
        sigma_max=model_args.sigma_max,
        sampling_eps=train_args.sampling_eps,
    )

    # Create the optimizer
    optim = torch.optim.Adam(unet.parameters(), lr=train_args.learning_rate)

    # Load the MNIST dataset
    dataset = torchvision.datasets.MNIST(
        root="~/.cache",
        train=True,
        download=True,
        transform=get_transforms(sample_size=model_args.sample_size),
    )
    # Create the data loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=train_args.batch_size,
        shuffle=train_args.shuffle,
        drop_last=train_args.dataloader_drop_last,
        num_workers=train_args.dataloader_num_workers,
    )

    # Train the model!
    train(
        train_args=train_args,
        valid_args=valid_args,
        unet=unet,
        noise_scheduler=noise_scheduler,
        optim=optim,
        data_loader=data_loader,
        device=device,
    )

    # Define the pipeline
    pipe = NCSNPipeline(unet=unet, scheduler=noise_scheduler)

    # Define a callback to decode the samples to grid image
    def decode_samples(
        pipe: NCSNPipeline,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict,
    ) -> Dict:
        # Get the samples from the callback kwargs
        # NOTE: The samples are cloned to avoid modifying the original
        samples = callback_kwargs["samples"]
        samples = samples.detach().clone()

        # Decode the samples to images
        samples = pipe.decode_samples(samples)
        images = pipe.numpy_to_pil(samples.cpu().numpy())

        # Create the grid image
        image = make_image_grid(
            images,
            rows=valid_args.num_grid_rows,
            cols=valid_args.num_grid_cols,
        )

        # Save the images
        image.save(
            train_args.animation_dir
            / f"timestep={timestep:03d}_annealing={step_index:03d}.png"
        )

        return callback_kwargs

    # Generate images with the pipeline; use `decode_samples` as a callback
    output = pipe(
        num_inference_steps=train_args.num_train_timesteps,
        batch_size=valid_args.num_generate_images,
        generator=torch.manual_seed(train_args.seed),
        callback_on_step_end=decode_samples,  # type: ignore
        callback_on_step_end_tensor_inputs=["samples"],
    )

    # Save the final image
    image = make_image_grid(
        images=output.images,  # type: ignore
        rows=valid_args.num_grid_rows,
        cols=valid_args.num_grid_cols,
    )
    image.save(train_args.output_dir / "final.png")

    # Save the trained model as a pipeline
    pipe.save_pretrained(train_args.output_dir / "ncsn-pipeline")

    # Push the pipeline to the hub (optional)
    # pipe.push_to_hub("py-img-gen/ncsn-mnist", private=True)


if __name__ == "__main__":
    parser = HfArgumentParser(dataclass_types=(ModelArgs, TrainArgs, ValidArgs))
    model_args, train_args, valid_args = parser.parse_args_into_dataclasses()
    main(model_args=model_args, train_args=train_args, valid_args=valid_args)
