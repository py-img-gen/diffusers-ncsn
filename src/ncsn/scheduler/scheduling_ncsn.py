import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import (
    ConfigMixin,
    register_to_config,
)
from diffusers.schedulers.scheduling_utils import (
    SchedulerMixin,
    SchedulerOutput,
)
from einops import rearrange


@dataclass
class AnnealedLangevinDynamicsOutput(SchedulerOutput):
    """Annealed Langevin Dynamics output class."""


class AnnealedLangevinDynamicsScheduler(SchedulerMixin, ConfigMixin):  # type: ignore
    """Annealed Langevin Dynamics scheduler for Noise Conditional Score Network (NCSN)."""

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int,
        num_annealed_steps: int,
        sigma_min: float,
        sigma_max: float,
        sampling_eps: float,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.num_annealed_steps = num_annealed_steps

        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sampling_eps = sampling_eps

        self._sigmas: Optional[torch.Tensor] = None
        self._step_size: Optional[torch.Tensor] = None
        self._timesteps: Optional[torch.Tensor] = None

        self.set_sigmas(num_inference_steps=num_train_timesteps)

    @property
    def sigmas(self) -> torch.Tensor:
        assert self._sigmas is not None
        return self._sigmas

    @property
    def step_size(self) -> torch.Tensor:
        assert self._step_size is not None
        return self._step_size

    @property
    def timesteps(self) -> torch.Tensor:
        assert self._timesteps is not None
        return self._timesteps

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        sampling_eps: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        sampling_eps = sampling_eps or self._sampling_eps
        self._timesteps = torch.arange(start=0, end=num_inference_steps)

    def set_sigmas(
        self,
        num_inference_steps: int,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        sampling_eps: Optional[float] = None,
    ) -> None:
        if self._timesteps is None:
            self.set_timesteps(
                num_inference_steps=num_inference_steps,
                sampling_eps=sampling_eps,
            )

        sigma_min = sigma_min or self._sigma_min
        sigma_max = sigma_max or self._sigma_max
        self._sigmas = torch.exp(
            torch.linspace(
                start=math.log(sigma_max),
                end=math.log(sigma_min),
                steps=num_inference_steps,
            )
        )

        sampling_eps = sampling_eps or self._sampling_eps
        self._step_size = sampling_eps * (self.sigmas / self.sigmas[-1]) ** 2

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[AnnealedLangevinDynamicsOutput, Tuple]:
        z = torch.randn_like(sample)
        step_size = self.step_size[timestep]
        sample = sample + 0.5 * step_size * model_output + torch.sqrt(step_size) * z

        if return_dict:
            return AnnealedLangevinDynamicsOutput(prev_sample=sample)
        else:
            return (sample,)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = timesteps.to(original_samples.device)
        sigmas = self.sigmas.to(original_samples.device)[timesteps]
        sigmas = rearrange(sigmas, "b -> b 1 1 1")
        noisy_samples = original_samples + noise * sigmas
        return noisy_samples
