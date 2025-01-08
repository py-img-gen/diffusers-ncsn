from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from einops import rearrange
from typing_extensions import Self

from .scheduler.scheduling_ncsn import (
    AnnealedLangevinDynamicsOutput,
    AnnealedLangevinDynamicsScheduler,
)
from .unet.unet_2d_ncsn import UNet2DModelForNCSN


def normalize_images(image: torch.Tensor) -> torch.Tensor:
    r"""Normalize the image to be between 0 and 1 using min-max normalization manner.

    Args:
        image (torch.Tensor):
            The batch of images to normalize.

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
        scheduler ([`AnnealedLangevinDynamicsScheduler`]):
            A `AnnealedLangevinDynamicsScheduler` to be used in combination with `unet` to estimate the score of the image.
    """

    unet: UNet2DModelForNCSN
    scheduler: AnnealedLangevinDynamicsScheduler

    _callback_tensor_inputs: List[str] = ["samples"]

    def __init__(
        self, unet: UNet2DModelForNCSN, scheduler: AnnealedLangevinDynamicsScheduler
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    def decode_samples(self, samples: torch.Tensor) -> torch.Tensor:
        r"""Decodes the generated samples to the correct format suitable for images.

        Args:
            samples (torch.Tensor):
                The generated samples to decode.

        Returns:
            torch.Tensor: The decoded samples.
        """
        # Normalize the generated image
        samples = normalize_images(samples)
        # Rearrange the generated image to the correct format
        samples = rearrange(samples, "b c w h -> b w h c")
        return samples

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 10,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[Self, int, int, Dict], Dict],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
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
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Returns:
            `diffusers.pipelines.ImagePipelineOutput` or `tuple`:
                If `return_dict` is `True`, `diffusers.pipelines.ImagePipelineOutput` is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        callback_on_step_end_tensor_inputs = (
            callback_on_step_end_tensor_inputs or self._callback_tensor_inputs
        )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        samples_shape = (
            batch_size,
            self.unet.config.in_channels,  # type: ignore
            self.unet.config.sample_size,  # type: ignore
            self.unet.config.sample_size,  # type: ignore
        )

        # Generate a random sample
        # NOTE: The behavior of random number generation is different between CPU and GPU,
        # so first generate random numbers on CPU and then move them to GPU (if available).
        sample = torch.rand(samples_shape, generator=generator)
        sample = sample.to(self.device)

        # Set the number of inference steps for the scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Perform the reverse diffusion process
        for t in self.progress_bar(self.scheduler.timesteps):
            # Perform `num_annnealed_steps` annealing steps
            for i in range(self.scheduler.num_annealed_steps):
                # Predict the score using the model
                model_output = self.unet(sample, t).sample  # type: ignore

                # Perform the annealed langevin dynamics
                output = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=sample,
                    generator=generator,
                    return_dict=return_dict,
                )
                sample = (
                    output.prev_sample
                    if isinstance(output, AnnealedLangevinDynamicsOutput)
                    else output[0]
                )

                # Perform the callback on step end if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]

                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    sample = callback_outputs.pop("samples", sample)

        sample = self.decode_samples(sample)

        if output_type == "pil":
            sample = self.numpy_to_pil(sample.cpu().numpy())

        if return_dict:
            return ImagePipelineOutput(images=sample)  # type: ignore
        else:
            return (sample,)
