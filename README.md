# SLTR-Net

Code for **SLTR-Net: Coordinated Readout of Spatial Semantics and Layer Trajectories for RGB-D Semantic Segmentation**, by Lishan Jia and Min Wang (CSAI 2026).

SLTR-Net uses dual-branch coupled semantic readout (DCSR) for spatial access and modality-conditioned layer-trajectory readout (MCLTR) for shared layer selection. The decoders retain separate feature updates and average their class probabilities for prediction.

## Setup

Use Python 3.11 or newer and a CUDA-enabled PyTorch 2.8 / torchvision 0.23 installation for the paper's evaluation protocol. Install the pinned backbone and this package from the repository root.

```bash
python -m pip install "rfdetr @ git+https://github.com/roboflow/rf-detr.git@e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe"
python -m pip install -e .
```

Download the [RF-DETR SegSmall initialisation checkpoint](https://github.com/roboflow/rf-detr/releases) `rf-detr-seg-small.pt`. The loader checks SHA-256 `6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3`. The six SLTR-Net checkpoints are distributed with the [versioned release](https://github.com/forever678-arch/SLTR-Net/releases/tag/v1.0.0). Each contains a tensor-only state dictionary. [Model zoo](sltrnet/model_zoo.json) lists the matching names, checksums and recorded results.

| Size | NYUv2 mIoU (%) | SUN RGB-D mIoU (%) |
|---|---:|---:|
| S | 60.19 | 53.79 |
| M | 61.38 | 54.75 |
| L | 62.68 | 55.32 |

The table reports the paper's five-scale results with flipping. The NYUv2 checkpoints average steps 2,000, 3,000 and 4,000. The SUN RGB-D checkpoints average steps 6,500, 7,500 and 8,500.

## Verify the checkpoint

The smoke command verifies the SHA-256 digests, loads all tensors strictly and runs one image-sized synthetic example through both branches.

```bash
python -m sltrnet.cli smoke --dataset nyuv2 --variant l --weights weights/sltr-net-nyuv2-l.pth --rf-checkpoint weights/rf-detr-seg-small.pt
```

## Evaluate

Prepare inputs using [the dataset instructions](docs/data_preparation.md). The two validity formats describe the original sensor observations separately from completed metric depth. The test data remain available from their original providers.

```bash
python -m sltrnet.cli evaluate --dataset nyuv2 --variant l --weights weights/sltr-net-nyuv2-l.pth --rf-checkpoint weights/rf-detr-seg-small.pt --data-root data/nyuv2 --valid-root data/nyuv2/nyuv2_raw_validity --scales 0.5 0.75 1.0 1.25 1.5 --flip --out results/nyuv2-l.json
python -m sltrnet.cli evaluate --dataset sunrgbd --variant l --weights weights/sltr-net-sunrgbd-l.pth --rf-checkpoint weights/rf-detr-seg-small.pt --data-root data/sunrgbd_processed --scales 0.5 0.75 1.0 1.25 1.5 --flip --out results/sunrgbd-l.json
```

Choose `--variant s` or `--variant m` with the corresponding checkpoint. For single-scale evaluation, use `--scales 1.0` and omit `--flip`.

## Train from RF-DETR initialisation

Prepare the training splits with [the dataset instructions](docs/data_preparation.md). For each dataset, `--variant s`, `m` or `l` selects the corresponding architecture. Both datasets start from the pinned RF-DETR checkpoint. The six released SLTR-Net weights are the results of training and averaging.

```bash
# NYUv2
python -m sltrnet.train fit --dataset nyuv2 --variant l --rf-checkpoint weights/rf-detr-seg-small.pt --data-root data/nyuv2 --valid-root data/nyuv2/nyuv2_raw_validity --run-dir runs/nyuv2-l --micro-batch 1

# SUN RGB-D
python -m sltrnet.train fit --dataset sunrgbd --variant l --rf-checkpoint weights/rf-detr-seg-small.pt --data-root data/sunrgbd_processed --run-dir runs/sunrgbd-l --micro-batch 1
```

The optimiser uses AdamW with seed 42, equal cross-entropy supervision for both heads and a nominal batch of 32 with the final partial batch retained. Every microbatch loss uses the number of valid pixels in the full loader batch as its denominator. Change `--micro-batch` to suit available video memory. CUDA forward passes use BF16 autocast and the loss is computed in FP32. The [training recipes](sltrnet/configs/training.json) specify the two learning rates, warm-up, cosine decay and gradient clipping.

| Dataset | LR schedule horizon | Default stop | Snapshots averaged |
|---|---:|---:|---|
| NYUv2 | 10,000 | 4,000 | 2,000 / 3,000 / 4,000 |
| SUN RGB-D | 16,000 | 9,000 | 6,500 / 7,500 / 8,500 |

`--stop-after` changes the stopping point while retaining the recipe's scheduler horizon. Each run directory contains the training settings and fixed-update model snapshots for averaging. Those snapshots store parameters for averaging and evaluation.

```bash
python -m sltrnet.train average --run-dir runs/nyuv2-l --out weights/nyuv2-l-avg.pth
python -m sltrnet.train average --run-dir runs/sunrgbd-l --out weights/sunrgbd-l-avg.pth
```

To evaluate a locally trained average, use `--local-weights` with the corresponding evaluation command and set `--weights` to the new average. The loader still checks the backbone checkpoint and matches every SLTR-Net tensor against the selected model. Use a new run directory with `fit --smoke` to check two synthetic training updates, snapshot writes and averaging without preparing a dataset.

## Licence

The project code uses the [MIT licence](LICENSE). The RF-DETR dependency and the underlying components retain their respective upstream licences. See [third-party notices](THIRD_PARTY_NOTICES.md).
