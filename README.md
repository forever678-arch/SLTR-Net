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

## Licence

The project code uses the [MIT licence](LICENSE). The RF-DETR dependency and the underlying components retain their respective upstream licences. See [third-party notices](THIRD_PARTY_NOTICES.md).
