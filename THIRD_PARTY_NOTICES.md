# Third-Party Notices

This project builds on the following third-party components. Per-file license
headers inside `seadpnet/rf/` are preserved verbatim.

## RF-DETR (roboflow/rf-detr)

- Used via the `rfdetr` Python package (pinned upstream commit
  `e96a71476bf09c738fa861c3d1ee2c8caeaa5ffe`, version string 1.11.0.dev0)
  as the backbone/runtime dependency, plus the vendored adapter package
  `seadpnet/rf/rfdetr_sem/` from this project's lineage.
- Upstream code and the `rf-detr-seg-small.pt` checkpoint are fetched from
  RF-DETR. The upstream distribution and licence are available at its project
  page.
- https://github.com/roboflow/rf-detr — https://roboflow.com/licensing

## DINOv2 (facebookresearch/dinov2)

- Indirect dependency through RF-DETR's backbone implementation. Apache-2.0.
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE

## Ultralytics YOLO11

- NOT a code dependency. The local CSP pyramid (`seadpnet/csp.py`) is a
  self-implementation inspired by the C3k2 split/concatenate design; it does
  not import the Ultralytics runtime.

## PyTorch / torchvision / transformers / timm

- BSD-style / Apache-2.0 licenses by their respective distributors; pinned
  versions in `pyproject.toml`.
