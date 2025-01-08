# 🤗 Noise Conditional Score Networks

[![CI](https://github.com/py-img-gen/diffusers-ncsn/actions/workflows/ci.yaml/badge.svg)](https://github.com/py-img-gen/diffusers-ncsn/actions/workflows/ci.yaml) 
[![Document](https://github.com/py-img-gen/diffusers-ncsn/actions/workflows/gh-pages.yaml/badge.svg)](https://github.com/py-img-gen/diffusers-ncsn/actions/workflows/gh-pages.yaml)
[![ermongroup/ncsn](https://img.shields.io/badge/Official_code-GitHub-green)](https://github.com/ermongroup/ncsn)
[![Model on HF](https://img.shields.io/badge/🤗%20Model%20on%20HF-py--img--gen/ncsn--mnist-D4AA00)](https://huggingface.co/py-img-gen/ncsn-mnist)

[`🤗 diffusers`](https://github.com/huggingface/diffusers) implementation of the paper ["Generative Modeling by Estimating Gradients of the Data Distribution" [Yang+ NeurIPS'19]](https://arxiv.org/abs/1907.05600).

## How to use

### Use without installation

You can load the pretrained pipeline directly from the HF Hub as follows:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import make_image_grid

# Specify the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# Load the pipeline from the Hugging Face Hub
#
pipe = DiffusionPipeline.from_pretrained(
    "py-img-gen/ncsn-mnist", trust_remote_code=True
)
pipe = pipe.to(device)

# Generate samples; here, we specify the seed and generate 16 images
output = pipe(
    batch_size=16,
    generator=torch.manual_seed(42),
)

# Create a grid image from the generated samples
image = make_image_grid(images=output.images, rows=4, cols=4)
image.save("output.png")
```

### Use with installation

First, install the package from this repository:

```shell
pip install git+https://github.com/py-img-gen/diffusers-ncsn
```

Then, you can use the package as follows:

```python
import torch

from ncsn.pipeline_ncsn import NCSNPipeline

# Specify the device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
# Load the pipeline from the HF Hub through the NCSNPipeline of this library
#
pipe = NCSNPipeline.from_pretrained("py-img-gen/ncsn-mnist", trust_remote_code=True)
pipe = pipe.to(device)

# Generate samples; here, we specify the seed and generate 16 images
output = pipe(
    batch_size=16,
    generator=torch.manual_seed(42),
)

# Create a grid image from the generated samples
image = make_image_grid(images=output.images, rows=4, cols=4)
image.save("output.png")
```

## Pretrained models and pipeline

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/py-img-gen/ncsn-mnist) 

## Showcase

### MNIST

Example of generating MNIST character images using the model trained with [`train_mnist.py`](https://github.com/py-img-gen/diffusers-ncsn/blob/main/train_mnist.py).

<p align="center">
    <img alt="mnist" src="https://github.com/user-attachments/assets/483b6637-2684-4844-8aa1-12b866d46226" width="50%" />
</p>

# Notes on uploading pipelines to the HF Hub with custom components

While referring to 📝 [Load community pipelines and components - huggingface diffusers](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview#community-components), pay attention to the following points.
- Change [the `_class_name` attribute](https://huggingface.co/py-img-gen/ncsn-mnist/blob/main/model_index.json#L2) in [`model_index.json`](https://huggingface.co/py-img-gen/ncsn-mnist/blob/main/model_index.json) to `["pipeline_ncsn", "NCSNPipeline"]`.
- Upload [`pipeline_ncsn.py`](https://github.com/py-img-gen/diffusers-ncsn/blob/main/src/ncsn/pipeline_ncsn.py) to [the root of the pipeline repository](https://huggingface.co/py-img-gen/ncsn-mnist/blob/main/pipeline_ncsn.py).
- Upload custom components to each subfolder:
  - Upload [`scheduling_ncsn.py`](https://github.com/py-img-gen/diffusers-ncsn/blob/main/src/ncsn/scheduler/scheduling_ncsn.py) to [the `scheduler` subfolder](https://huggingface.co/py-img-gen/ncsn-mnist/tree/main/scheduler).
  - Upload [`unet_2d_ncsn.py`](https://github.com/py-img-gen/diffusers-ncsn/blob/main/src/ncsn/unet/unet_2d_ncsn.py) to [the `unet` subfolder](https://huggingface.co/py-img-gen/ncsn-mnist/tree/main/unet).
- Ensure that the custom components are placed in each subfolder because they are referenced by relative paths from `pipeline_ncsn.py`.
  - Based on this, the code in this library is also placed in the same directory structure as the HF Hub.
  - For example, `pipeline_ncsn.py` imports `unet_2d_ncsn.py` as `from .unet.unet_2d_ncsn import UNet2DModelForNCSN` because it is placed in the `unet` subfolder.

## Acknowledgements

- JeongJiHeon/ScoreDiffusionModel: The Pytorch Tutorial of Score-based and Diffusion Model https://github.com/JeongJiHeon/ScoreDiffusionModel/tree/main 
