from typing import Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from einops import rearrange

from .scheduling_ncsn import (
    AnnealedLangevinDynamicOutput,
    AnnealedLangevinDynamicScheduler,
)
from .unet_2d_ncsn import UNet2DModelForNCSN


def normalize_images(image: torch.Tensor) -> torch.Tensor:
    """Normalize the image to be between 0 and 1 using min-max normalization manner.

    Args:
        image (torch.Tensor): The batch of images to normalize.

    Returns:
        torch.Tensor: The normalized image.
    """
    assert image.ndim == 4, image.ndim
    batch_size = image.shape[0]

    def _normalize(img: torch.Tensor) -> torch.Tensor:
        return (img - img.min()) / (img.max() - img.min())

    for i in range(batch_size):
        image[i] = _normalize(image[i])
    return image


class NCSNPipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation using Noise Conditional Score Network (NCSN).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModelForNCSN`]):
            A `UNet2DModelForNCSN` to estimate the score of the image.
        scheduler ([`AnnealedLangevinDynamicScheduler`]):
            A `AnnealedLangevinDynamicScheduler` to be used in combination with `unet` to estimate the score of the image.
    """

    unet: UNet2DModelForNCSN
    scheduler: AnnealedLangevinDynamicScheduler

    def __init__(
        self, unet: UNet2DModelForNCSN, scheduler: AnnealedLangevinDynamicScheduler
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 10,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 10):
                The number of inference steps.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        sample_shape = (
            batch_size,
            self.unet.config.in_channels,  # type: ignore
            self.unet.config.sample_size,  # type: ignore
            self.unet.config.sample_size,  # type: ignore
        )
        # Generate a random sample
        sample = torch.rand(sample_shape, generator=generator)
        sample = sample.to(self.device)

        # Set the number of inference steps for the scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Perform the reverse diffusion process
        for t in self.progress_bar(self.scheduler.timesteps):
            # Predict the score using the model
            model_output = self.unet(sample, t).sample  # type: ignore

            # Perform the annealed langevin dynamics
            output = self.scheduler.step(
                model_output=model_output,
                model=self.unet,
                timestep=t,
                sample=sample,
                generator=generator,
                return_dict=return_dict,
            )
            sample = (
                output.prev_sample
                if isinstance(output, AnnealedLangevinDynamicOutput)
                else output[0]
            )

        # Normalize the generated image
        sample = normalize_images(sample)

        # Rearrange the generated image to the correct format
        sample = rearrange(sample, "b c w h -> b w h c")

        if output_type == "pil":
            sample = self.numpy_to_pil(sample.cpu().numpy())

        if return_dict:
            return ImagePipelineOutput(images=sample)  # type: ignore
        else:
            return (sample,)
