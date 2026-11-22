"""Load the paper checkpoints and run smoke checks or semantic evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
RF_SHA256 = '6de3da31b2572cac214a1c76cce4a92a13966d56390ac2b3a3de9a8dc2b2bca3'


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_config(dataset, variant):
    name = f'{dataset}-{variant.lower()}'
    return json.loads((ROOT / 'configs' / f'{name}.json').read_text(encoding='utf-8'))


def build_model(dataset, variant, config, backbone):
    if digest_file(backbone) != RF_SHA256:
        raise ValueError('Unexpected RF-DETR SegSmall checkpoint')
    config = {**config, 'pretrained_checkpoint': str(Path(backbone).resolve())}
    if dataset == 'nyuv2':
        from .family_model import FamilyH

        model = FamilyH(config, f'SLTR_{variant.upper()}')
    elif variant.lower() == 'l':
        from .sun_reference_model import SunReadoutH

        model = SunReadoutH(config)
    else:
        from .sun_model import SunFamilyH

        model = SunFamilyH(config)
    name = f'{dataset}-{variant.lower()}'
    record = json.loads((ROOT / 'model_zoo.json').read_text(encoding='utf-8'))[name]
    if sum(p.numel() for p in model.parameters()) != record['parameters']:
        raise ValueError(f'Parameter count mismatch for {name}')
    return model, record


def load_model(dataset, variant, weights, backbone, device, *, local_weights=False):
    config = load_config(dataset, variant)
    model, record = build_model(dataset, variant, config, backbone)
    if not local_weights and digest_file(weights) != record['sha256']:
        raise ValueError('Unexpected release checkpoint. For a locally trained tensor-only average, pass --local-weights.')
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), config


@torch.inference_mode()
def smoke(model, config, device):
    height, width = config['input_size']
    padded = ((height + 23) // 24 * 24, (width + 23) // 24 * 24)
    pad = (0, padded[1] - width, 0, padded[0] - height)
    generator = torch.Generator().manual_seed(42)
    rgb = torch.rand((1, 3, height, width), generator=generator).to(device)
    depth = torch.rand((1, 1, height, width), generator=generator).to(device)
    valid = (depth > 0.15).float()
    mask = torch.ones((1, *padded), device=device, dtype=torch.bool)
    mask[:, :height, :width] = False
    outputs = model(F.pad(rgb, pad), mask, F.pad(depth, pad), F.pad(valid, pad))
    prediction = []
    for name in ('semantic_logits', 'semantic_logits_b'):
        logits = outputs[name][..., :height, :width]
        if logits.shape != (1, config['num_classes'], height, width) or not torch.isfinite(logits).all():
            raise RuntimeError(f'Invalid logits for {name}')
        prediction.append(logits.softmax(1))
    result = (0.5 * (prediction[0] + prediction[1])).argmax(1)
    return {'status': 'pass', 'model': config['model_name'], 'prediction_shape': list(result.shape)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'evaluate'))
    parser.add_argument('--dataset', choices=('nyuv2', 'sunrgbd'), required=True)
    parser.add_argument('--variant', choices=('s', 'm', 'l'), required=True)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--local-weights', action='store_true', help='Load a locally averaged state dictionary with strict model matching')
    parser.add_argument('--rf-checkpoint', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--valid-root', type=Path, help='NYUv2 original sensor validity maps')
    parser.add_argument('--scales', type=float, nargs='+', default=(1.0,))
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'evaluate':
        if not args.data_root or (args.dataset == 'nyuv2' and not args.valid_root):
            parser.error('evaluate requires --data-root and, for NYUv2, --valid-root')
    model, config = load_model(args.dataset, args.variant, args.weights, args.rf_checkpoint, args.device,
                               local_weights=args.local_weights)
    if args.command == 'smoke':
        result = smoke(model, config, args.device)
    else:
        if args.dataset == 'nyuv2':
            from seadpnet.evaluation.nyuv2 import evaluate
        else:
            from seadpnet.evaluation.sunrgbd import evaluate
        config['data_root'] = str(args.data_root.resolve())
        if args.valid_root:
            config['valid_root'] = str(args.valid_root.resolve())
        result = evaluate(model, config, split=config.get('test_split', 'test'), scales=args.scales, flip=args.flip,
                          batch_size=1, device=args.device)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
