# 🤗 Noise Conditional Score Networks

[![CI](https://github.com/py-img-gen/diffusers-ncsn/actions/workflows/ci.yaml/badge.svg)](https://github.com/py-img-gen/diffusers-ncsn/actions/workflows/ci.yaml) 
[![](https://img.shields.io/badge/Official_code-GitHub-green)](https://github.com/ermongroup/ncsn)
[![Model on HF](https://img.shields.io/badge/🤗%20Model%20on%20HF-py--img--gen/ncsn--mnist-D4AA00)](https://huggingface.co/py-img-gen/ncsn-mnist)

[`🤗 diffusers`](https://github.com/huggingface/diffusers) implementation of the paper ["Generative Modeling by Estimating Gradients of the Data Distribution" [Yang+ NeurIPS'19]](https://arxiv.org/abs/1907.05600).

## Installation

```shell
pip install git+https://github.com/py-img-gen/diffusers-ncsn
```

## Pretrained models and pipeline

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/py-img-gen/ncsn-mnist)

## Showcase

### MNIST

Example of generating MNIST character images using the model trained with [`train_mnist.py`](https://github.com/py-img-gen/diffusers-ncsn/blob/main/train_mnist.py).

<div align="center">
    <img alt="mnist" src="https://github.com/user-attachments/assets/483b6637-2684-4844-8aa1-12b866d46226" width="50%" />
</div>

## Acknowledgements

- JeongJiHeon/ScoreDiffusionModel: The Pytorch Tutorial of Score-based and Diffusion Model https://github.com/JeongJiHeon/ScoreDiffusionModel/tree/main 
