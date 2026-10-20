"""Load a saved checkpoint and evaluate held-out ESC-50 fold 5."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from esc50 import (ROOT, SPLITS, AudioNet, Features, environment, evaluate, load_features,
                   make_loader, metadata, save_test_results, setup)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/ESC-50-master')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/baseline/audionet.ckpt')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'data/cache')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run train.py first.')
    device = setup(device=args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint['splits'] != SPLITS:
        raise ValueError('Unsupported checkpoint format or split configuration.')
    meta = metadata(args.data_dir)
    classes = meta.sort_values('target').drop_duplicates('target').category.tolist()
    if classes != checkpoint['classes']:
        raise ValueError('Dataset class mapping differs from the training checkpoint.')
    meta_hash = hashlib.sha256((args.data_dir/'meta/esc50.csv').read_bytes()).hexdigest()
    if meta_hash != checkpoint['run_config']['metadata_sha256']:
        raise ValueError('Dataset metadata differs from the training checkpoint.')
    model = AudioNet(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    x, _ = load_features(args.data_dir, meta, Features(**checkpoint['features']), args.cache_dir)
    loss, accuracy, truth, probabilities = evaluate(model, make_loader(meta, x, 'test', args.batch_size), device)
    metrics = dict(test_fold=5, samples=len(truth), accuracy=accuracy, loss=loss,
                   selected_epoch=checkpoint['epoch'], validation_accuracy=checkpoint['validation_accuracy'],
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   environment=environment(device), chance_accuracy=.02)
    output = args.output_dir or args.checkpoint.parent/'test'
    save_test_results(meta, truth, probabilities, classes, metrics, output)
    print(json.dumps(metrics, indent=2))
    print(f'Test results saved to {output.resolve()}')


if __name__ == '__main__':
    main()
