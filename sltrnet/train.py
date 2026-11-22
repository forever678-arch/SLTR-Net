"""Train SLTR-Net with the paper schedules, or average checkpoints from one run."""

import argparse
import itertools
import json
from pathlib import Path
import uuid

import torch

from seadpnet.data.nyuv2 import forward_batch
from seadpnet.evaluation.common import amp_context
from seadpnet.losses import dual_ce, split_batch
from seadpnet.runtime import optimizer_for, save_checkpoint, seed_all, write_json
from seadpnet.schedule import lr_multiplier, run_updates
from .cli import ROOT, build_model, load_config


def training_batches(dataset, config):
    if dataset == 'nyuv2':
        from seadpnet.data.nyuv2 import loader

        batches = loader(config, config['train_split'], augment=True, batch_size=config['loader_batch'])
        if len(batches.dataset) != config['train_images']:
            raise ValueError('NYUv2 training split must have 795 images')
        # Reusing the loader advances its independent sampler generator each epoch.
        while True:
            yield from batches
    else:
        from seadpnet.data.sunrgbd import loader

        for epoch in itertools.count():
            batches = loader(config, config['train_split'], augment=True,
                             batch_size=config['loader_batch'], epoch=epoch)
            if len(batches.dataset) != config['train_images']:
                raise ValueError('SUN RGB-D training split must have 5,285 images')
            yield from batches


def synthetic_batches(classes):
    generator = torch.Generator().manual_seed(42)
    for _ in itertools.count():
        depth = torch.rand((2, 1, 96, 96), generator=generator)
        yield {
            'image': torch.randn((2, 3, 96, 96), generator=generator),
            'depth': depth, 'valid': (depth > 0.15).float(),
            'padding_mask': torch.zeros((2, 96, 96), dtype=torch.bool),
            'semantic': torch.randint(classes, (2, 96, 96), generator=generator),
        }


def train(args):
    if not args.smoke and (not args.data_root or (args.dataset == 'nyuv2' and not args.valid_root)):
        raise ValueError('Training requires --data-root and, for NYUv2, --valid-root')
    recipe = json.loads((ROOT / 'configs' / 'training.json').read_text(encoding='utf-8'))[args.dataset]
    config = {**load_config(args.dataset, args.variant), **recipe,
              'num_workers': args.num_workers, 'data_root': str(args.data_root), 'valid_root': str(args.valid_root)}
    if not 1 <= args.micro_batch <= recipe['loader_batch']:
        raise ValueError('--micro-batch must be between 1 and 32')
    if args.smoke and args.stop_after not in (None, 2):
        raise ValueError('The synthetic training smoke test always runs two updates')
    requested_stop = recipe['stop_updates'] if args.stop_after is None else args.stop_after
    stop = run_updates(recipe['train_updates'], 2 if args.smoke else requested_stop)
    snapshots = [1, 2] if args.smoke else list(recipe['average_updates'])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    run = {'run_id': uuid.uuid4().hex, 'dataset': args.dataset, 'variant': args.variant,
           'recipe': recipe, 'actual_stop': stop, 'average_updates': snapshots,
           'micro_batch': args.micro_batch, 'synthetic_smoke': args.smoke}
    write_json(args.run_dir / 'run.json', run)
    seed_all(recipe['seed'])
    model, _ = build_model(args.dataset, args.variant, config, args.rf_checkpoint)
    model = model.to(args.device).train()
    optimizer = optimizer_for(model, recipe)

    def multiplier(step):
        return lr_multiplier(step, recipe['train_updates'], recipe['warmup_updates'], recipe['min_lr_ratio'])

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    batches = synthetic_batches(config['num_classes']) if args.smoke else training_batches(args.dataset, config)
    for step, batch in enumerate(itertools.islice(batches, stop), 1):
        total_valid = int((batch['semantic'] != 255).sum())
        if total_valid == 0:
            raise ValueError(f'Training batch {step} has no labelled pixels')
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for part in split_batch(batch, args.micro_batch):
            labels = part['semantic'].to(args.device)
            with amp_context(args.device):
                prediction = forward_batch(model, part, args.device)
            loss, _ = dual_ce(prediction, labels, total_valid)
            loss.backward()
            loss_sum += float(loss.detach())
            del loss, prediction, labels
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step in snapshots or step == stop:
            state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            save_checkpoint(args.run_dir / f'u{step}.pth', {'run_id': run['run_id'], 'update': step, 'model': state})
        if step == 1 or step % 100 == 0 or step == stop:
            print(json.dumps({'update': step, 'images': len(batch['semantic']), 'loss': loss_sum,
                              'gradient_norm': float(norm), 'next_learning_rates': scheduler.get_last_lr()}), flush=True)


def average(run_dir, out):
    run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    updates = run['average_updates']
    if not updates or sorted(set(updates)) != updates:
        raise ValueError('Expected an ordered list of distinct averaging updates')
    if out.exists():
        raise FileExistsError(out)
    states = []
    for step in updates:
        checkpoint = torch.load(run_dir / f'u{step}.pth', map_location='cpu', weights_only=True)
        if checkpoint['run_id'] != run['run_id'] or checkpoint['update'] != step:
            raise ValueError(f'Checkpoint u{step} belongs to a different run or update')
        states.append(checkpoint['model'])
    if any(state.keys() != states[0].keys() for state in states[1:]):
        raise ValueError('Snapshot parameter names differ')
    result = {}
    for name, first in states[0].items():
        values = [state[name] for state in states]
        if any(value.shape != first.shape or value.dtype != first.dtype for value in values):
            raise ValueError(f'Snapshot tensor is incompatible: {name}')
        if first.is_floating_point():
            aggregate = torch.zeros_like(first, dtype=torch.float64)
            for value in values:
                aggregate.add_(value.double())
            result[name] = aggregate.div_(len(values)).to(first.dtype)
        else:
            if any(not torch.equal(first, value) for value in values[1:]):
                raise ValueError(f'Nonfloating buffer differs: {name}')
            result[name] = first
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out, result)
    print(f'Averaged updates {updates} into {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Train from the pinned RF-DETR initialisation')
    fit.add_argument('--dataset', choices=('nyuv2', 'sunrgbd'), required=True)
    fit.add_argument('--variant', choices=('s', 'm', 'l'), required=True)
    fit.add_argument('--rf-checkpoint', type=Path, required=True)
    fit.add_argument('--run-dir', type=Path, required=True, help='New output directory')
    fit.add_argument('--data-root', type=Path)
    fit.add_argument('--valid-root', type=Path)
    fit.add_argument('--micro-batch', type=int, default=1)
    fit.add_argument('--num-workers', type=int, default=4)
    fit.add_argument('--device', default='cuda')
    fit.add_argument('--stop-after', type=int, help='Stop early while retaining the original scheduler horizon')
    fit.add_argument('--smoke', action='store_true', help='Run two 96×96 synthetic training updates')
    mean = commands.add_parser('average', help='Average the prescribed snapshots from one run')
    mean.add_argument('--run-dir', type=Path, required=True)
    mean.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fit':
        train(args)
    else:
        average(args.run_dir, args.out)


if __name__ == '__main__':
    main()
