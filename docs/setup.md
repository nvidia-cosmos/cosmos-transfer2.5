# Setup Guide

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver >=570.124.06 compatible with [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.31 (e.g Ubuntu >=22.04)
* Python 3.10

## Installation

Clone the repository:

```bash
git clone git@github.com:nvidia-cosmos/cosmos-transfer2.git
cd cosmos-transfer2
```

Installing system dependencies:

Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package into a new environment:

```shell
uv sync
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
uv sync --active --inexact
```

## Downloading Checkpoints

Checkpoints are automatically downloaded during inference and post-training. To modify the checkpoint cache location, set the [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) environment variable.

### Troubleshooting

* **CUDA driver version insufficient**: Update NVIDIA drivers to latest version compatible with CUDA 12.8.1+

Check driver compatibility:

```shell
nvidia-smi | grep "CUDA Version:"
```

* **Out of Memory (OOM) errors**: Use 2B models instead of 14B, multi-GPU, or reduce batch size/resolution

For other issues, check [GitHub Issues](https://github.com/nvidia-cosmos/cosmos-transfer2.5/issues).

## Advanced

### Docker container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed. To avoid running out of file descriptors when building the container, increase the limit with `--ulimit nofile` as in the example below. 

Example build command:

```bash
docker build --ulimit nofile=131071:131071 -f Dockerfile . -t cosmos-transfer-2.5
```

Example run command:

```bash
docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it cosmos-transfer-2.5
```
