# Diffusers Inference Guide

This guide explains how to use `scripts/diffusers_inference.py` to run
Cosmos-Transfer2.5 Diffusers pipelines for single-control generation
(edge, visual blur, depth, or segmentation). Review the [Inference Guide](inference.md)
for broader context.

## Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), via:
   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Ensure example assets are available under `assets/` (for example `assets/seg.jsonl`,
   `assets/robot_example/*`, and `assets/car_example/*`).

## Script Overview

`scripts/diffusers_inference.py` runs
[`Cosmos2_5_TransferPipeline`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos#diffusers.Cosmos2_5_TransferPipeline)
with one [`CosmosControlNetModel`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos#diffusers.CosmosControlNetModel)
control stream at a time.

Provide sample configs through `--input-files` using `.json` or `.jsonl` files. If multiple rows are present,
`--sample-index` selects which row to run. Paths inside sample files (`prompt_path`, `video_path`, and
nested `control_path`) are resolved relative to the config file location.

CLI overrides are supported for sample fields (for example prompt, guidance, steps, and resolution).

| Flag | Purpose |
| --- | --- |
| `--input-files`, `-i` | One or more `.json` / `.jsonl` sample files. |
| `--sample-index` | Selects which sample row to execute when loading a `.jsonl` with multiple rows. |
| `--control-key` | Explicitly selects control modality: `edge`, `vis`, `depth`, or `seg`. |
| `--output-path`, `-o` | Output file path or output directory. If a directory is provided, output filename defaults to `{sample.name}.mp4` (or `.jpg` for single-frame output). |
| `--output-fps` | FPS used when exporting generated/control videos. |
| `--force-generate-controls` | Recomputes controls on the fly even if `control_path` is present in config. |
| `--save-controls` | Saves the resolved control stream next to output as `*_control.mp4` or `*_control.jpg`. |
| `--model-id` / `--revision` | Base model and revision (defaults: `nvidia/Cosmos-Transfer2.5-2B`, `diffusers/general`). |
| `--controlnet-revision-template` | ControlNet revision pattern (default: `diffusers/controlnet/general/{control_key}`). |
| `--device`, `--device-map`, `--torch-dtype` | Device placement and numeric precision controls. |
| `--mock-safety-checker` | Bypasses guardrails with a mock checker (use for debugging only). |
| `--seg-mode` | Segmentation control generation mode: `auto` (default) or `point`. |
| `--large-depth-model` / `--sam2-model-id` | Use large Video Depth Anything weights for on-the-fly depth generation (default uses small); override SAM2 model ID for segmentation generation. |

Run `./scripts/diffusers_inference.py --help` to see the full options

## Ready-to-Run Asset Examples

Run these commands from `packages/cosmos-transfer2`.

>[!IMPORTANT]
>These examples pass in --mock-safety-checker, but please do not abuse this. Refer to the license you signed on HuggingFace for the agreement you signed.


### Segmentation

```bash
./scripts/diffusers_inference.py \
  -o ./outputs/diffusers/seg \
  --input-files assets/seg.jsonl \
  --control-key seg \
  --sample-index 0 \
  --output-fps 30 \
  --save-controls \
  --force-generate-controls \
  --mock-safety-checker
```

### Visual Blur (`vis`)

```bash
./scripts/diffusers_inference.py \
  -o ./outputs/diffusers/vis \
  --input-files assets/vis.jsonl \
  --control-key vis \
  --sample-index 0 \
  --output-fps 30 \
  --save-controls \
  --force-generate-controls \
  --mock-safety-checker
```

### Depth

```bash
./scripts/diffusers_inference.py \
  -o ./outputs/diffusers/depth \
  --input-files assets/depth.jsonl \
  --control-key depth \
  --sample-index 0 \
  --output-fps 30 \
  --save-controls \
  --force-generate-controls \
  --mock-safety-checker
```

### Edge

```bash
./scripts/diffusers_inference.py \
  -o ./outputs/diffusers/edge \
  --input-files assets/edge.jsonl \
  --control-key edge \
  --sample-index 0 \
  --output-fps 30 \
  --save-controls \
  --force-generate-controls \
  --mock-safety-checker
```

## Tips

- `assets/*.jsonl` includes multiple samples. Use `--sample-index 0` for robot examples and `--sample-index 1` for car examples.
- Diffusers Transfer2 currently supports exactly one control modality per run. Avoid passing sample rows that contain multiple active controls.
- For `seg` control, `--seg-mode point` uses a single point seed (from `seg.control_prompt` coordinates if provided), while `--seg-mode auto` seeds many points and tracks multiple regions.
- Set `--save-controls` when debugging computed controls, especially for on-the-fly depth/seg generation.
