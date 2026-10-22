"""Train on folds 1-3 and select the best checkpoint using fold 4 only."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from esc50 import (ROOT, SPLITS, Features, augment_batch, build_model, environment, evaluate, load_features,
                   make_loader, metadata, plot_example, plot_training, setup, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/ESC-50-master')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/baseline')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'data/cache')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--scheduler', choices=['none', 'cosine'], default='none')
    parser.add_argument('--feature-preset', choices=['standardized', 'original', 'highres'], default='standardized',
                        help='original reproduces notebook mel defaults without standardization')
    parser.add_argument('--architecture', choices=['baseline','spectrogram'],default='baseline')
    parser.add_argument('--augment', action='store_true', help='Training-only time shifts and spectrogram masking')
    parser.add_argument('--mixup',type=float,default=0.,help='Beta distribution alpha; zero disables mixup')
    parser.add_argument('--label-smoothing',type=float,default=0.)
    parser.add_argument('--optimizer',choices=['adam','adamw'],default='adam')
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error('Use positive epochs, batch size, learning rate; nonnegative weight decay; dropout in [0,1).')
    if args.mixup < 0 or not 0 <= args.label_smoothing < 1:
        parser.error('Mixup alpha must be nonnegative; label smoothing must be in [0,1).')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f'Output directory is not empty: {out}. Choose a new --output-dir to preserve previous runs.')
    device = setup(args.seed, args.device)
    meta = metadata(args.data_dir)
    presets = {'standardized':Features(), 'original':Features(n_mels=128,n_fft=400,hop_length=200,normalize=False),
               'highres':Features(sample_rate=16000,n_mels=64,n_fft=1024,hop_length=320)}
    config = presets[args.feature_preset]
    x, signature = load_features(args.data_dir, meta, config, args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_config = dict(n_classes=50, base_filters=32, dropout=args.dropout)
    if args.architecture != 'baseline':
        model_config['architecture'] = args.architecture
    classes = meta.sort_values('target').drop_duplicates('target').category.tolist()
    run = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run.update(features=asdict(config), model=model_config, splits=SPLITS, environment=environment(device),
               dataset_signature=signature, metadata_sha256=hashlib.sha256((args.data_dir/'meta/esc50.csv').read_bytes()).hexdigest())
    write_json(out/'run_config.json', run)
    plot_example(args.data_dir, meta.iloc[0], x[0], out)
    model = build_model(model_config).to(device)
    optimizer_class = torch.optim.AdamW if args.optimizer=='adamw' else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.scheduler=='cosine' else None
    train = make_loader(meta, x, 'train', args.batch_size, args.seed)
    val = make_loader(meta, x, 'validation', args.batch_size)
    print(f'Device: {device}; train: {len(train.dataset)}; validation: {len(val.dataset)}', flush=True)
    history, best, selected = [], -1.0, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total, correct, count = 0., 0, 0
        for xb, y in train:
            xb, y = xb.to(device), y.to(device)
            if args.augment:
                xb = augment_batch(xb)
            lam = float(np.random.beta(args.mixup,args.mixup)) if args.mixup else 1.
            permutation = torch.randperm(len(y),device=device) if args.mixup else torch.arange(len(y),device=device)
            if args.mixup:
                xb = lam*xb+(1-lam)*xb[permutation]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = lam*torch.nn.functional.cross_entropy(logits,y,label_smoothing=args.label_smoothing)
            if args.mixup:
                loss = loss+(1-lam)*torch.nn.functional.cross_entropy(logits,y[permutation],label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(y)
            correct += (lam*(logits.argmax(1)==y).sum()+(1-lam)*(logits.argmax(1)==y[permutation]).sum()).item()
            count += len(y)
        vl, accuracy, _, _ = evaluate(model, val, device)
        row = dict(epoch=epoch, train_loss=total/count, train_accuracy=correct/count,
                   val_loss=vl, val_accuracy=accuracy, learning_rate=optimizer.param_groups[0]['lr'])
        history.append(row)
        print(f'Epoch {epoch:02}/{args.epochs}: train loss {total/count:.4f}, validation accuracy {accuracy:.2%}', flush=True)
        if accuracy > best:
            best, selected = accuracy, epoch
            torch.save(dict(format_version=1, state_dict=model.state_dict(), model_config=model_config,
                            features=asdict(config), classes=classes, splits=SPLITS, epoch=epoch,
                            validation_accuracy=accuracy, run_config=run), out/'audionet.ckpt')
        pd.DataFrame(history).to_csv(out/'history.csv', index=False)
        if scheduler:
            scheduler.step()
    plot_training(pd.DataFrame(history), out)
    write_json(out/'train_metrics.json', dict(epochs=args.epochs, selected_epoch=selected,
                                             best_validation_accuracy=best, train_samples=count,
                                             validation_samples=len(val.dataset)))
    print(f'Best validation accuracy: {best:.2%} (epoch {selected}). Saved {out/"audionet.ckpt"}', flush=True)


if __name__ == '__main__':
    main()
