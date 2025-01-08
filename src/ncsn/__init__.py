from importlib.metadata import version

from ncsn.pipeline_ncsn import NCSNPipeline

__version__ = version("diffusers-ncsn")


__all__ = [
    "NCSNPipeline",
]
