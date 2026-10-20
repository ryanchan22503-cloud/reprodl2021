"""Shared ESC-50 data, model, and evaluation code. Original AudioNet: Simone Scardapane, MIT."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
import torchaudio

ROOT = Path(__file__).resolve().parent
SPLITS = {'train': [1, 2, 3], 'validation': [4], 'test': [5]}


@dataclass(frozen=True)
class Features:
    sample_rate: int = 8000
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 256
    top_db: int = 80
    normalize: bool = True


def setup(seed=0, device='auto'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu or install a compatible GPU build.')
    return torch.device(device)


def metadata(data_dir):
    data_dir = Path(data_dir)
    csv = data_dir / 'meta/esc50.csv'
    if not csv.is_file():
        raise FileNotFoundError(f'Metadata not found: {csv}. Set --data-dir to the folder containing audio/ and meta/.')
    meta = pd.read_csv(csv)
    required = {'filename', 'fold', 'target', 'category'}
    if not required.issubset(meta.columns):
        raise ValueError(f'Metadata must contain {sorted(required)}')
    if len(meta) != 2000 or set(meta.fold) != set(range(1, 6)) or set(meta.target) != set(range(50)):
        raise ValueError('Expected complete ESC-50 metadata: 2,000 clips, five folds, 50 classes.')
    if meta.filename.duplicated().any() or not (meta.groupby(['fold', 'target']).size() == 8).all():
        raise ValueError('Duplicate filenames or incorrect class counts in folds.')
    if meta.groupby('target').category.nunique().ne(1).any() or meta.groupby('category').target.nunique().ne(1).any():
        raise ValueError('Class labels and names must map one to one.')
    for name in meta.filename:
        if Path(name).name != name or not (data_dir/'audio'/name).is_file():
            raise FileNotFoundError(f'Missing or invalid audio file: {name}')
    return meta


def dataset_signature(data_dir, meta):
    """Invalidate cached features after metadata/audio changes (size and modification time)."""
    digest = hashlib.sha256((Path(data_dir)/'meta/esc50.csv').read_bytes())
    for name in meta.filename:
        stat = (Path(data_dir)/'audio'/name).stat()
        digest.update(f'{name}:{stat.st_size}:{stat.st_mtime_ns}'.encode())
    return digest.hexdigest()


def feature_key(signature, config):
    payload = {'version': 1, 'dataset': signature, 'features': asdict(config),
               'torch': str(torch.__version__), 'torchaudio': str(torchaudio.__version__)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def load_features(data_dir, meta, config, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = dataset_signature(data_dir, meta)
    cache = cache_dir/f'{feature_key(signature, config)}.pt'
    if cache.is_file():
        x = torch.load(cache, weights_only=True)
        if x.ndim != 4 or x.shape[:3] != (len(meta), 1, config.n_mels) or not torch.isfinite(x).all():
            raise ValueError(f'Invalid feature cache; remove {cache} and retry.')
        print(f'Using cached features: {cache}', flush=True)
        return x, signature
    resamplers = {}
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate, n_fft=config.n_fft,
                                               hop_length=config.hop_length, n_mels=config.n_mels)
    db = torchaudio.transforms.AmplitudeToDB(top_db=config.top_db)
    xs = []
    for i, row in meta.iterrows():
        wav, sr = sf.read(Path(data_dir)/'audio'/row.filename, dtype='float32', always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1)).unsqueeze(0)
        if sr != config.sample_rate:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, config.sample_rate)
            wav = resamplers[sr](wav)
        # ESC-50 clips are five seconds. Explicit length keeps every batch stackable.
        length = config.sample_rate * 5
        wav = nn.functional.pad(wav[..., :length], (0, max(0, length-wav.shape[-1])))
        spec = db(mel(wav))
        if config.normalize:
            spec = (spec-spec.mean())/spec.std().clamp_min(1e-6)
        xs.append(spec)
        if (i+1) % 200 == 0:
            print(f'Features: {i+1}/{len(meta)}', flush=True)
    x = torch.stack(xs)
    if not torch.isfinite(x).all():
        raise ValueError('Non-finite features in audio data.')
    torch.save(x, cache)
    return x, signature


class AudioNet(nn.Module):
    """Original four-convolution architecture, with optional head dropout."""
    def __init__(self, n_classes=50, base_filters=32, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(base_filters, base_filters*2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters*2)
        self.conv4 = nn.Conv2d(base_filters*2, base_filters*4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters*4)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(base_filters*4, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool2(torch.relu(self.bn4(self.conv4(x))))
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc1(self.dropout(x))


def make_loader(meta, x, split, batch_size, seed=0):
    mask = torch.tensor(meta.fold.isin(SPLITS[split]).to_numpy())
    labels = torch.tensor(meta.target.to_numpy(), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x[mask], labels[mask])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=split=='train',
                                       generator=torch.Generator().manual_seed(seed), num_workers=0)


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    total, truth, probabilities = 0.0, [], []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        truth.append(y.cpu())
        probabilities.append(logits.softmax(1).cpu())
    y = torch.cat(truth).numpy()
    p = torch.cat(probabilities).numpy()
    return total/len(y), float((p.argmax(1)==y).mean()), y, p


def environment(device):
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchaudio': str(torchaudio.__version__), 'numpy': np.__version__,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type=='cuda' else None}


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2), encoding='utf-8')


def plot_training(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ['loss', 'accuracy'], ['Loss', 'Accuracy']):
        ax.plot(history.epoch, history['train_'+metric], label='Train (folds 1-3)')
        ax.plot(history.epoch, history['val_'+metric], label='Validation (fold 4)')
        ax.set(xlabel='Epoch', ylabel=label, title=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output)/'training_curves.png', dpi=160)
    plt.close(fig)


def plot_example(data_dir, row, x, output):
    wav, sr = sf.read(Path(data_dir)/'audio'/row.filename)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(len(wav))/sr, wav, linewidth=.4)
    axes[0].set(title=f'Example: {row.category}', xlabel='Time (seconds)', ylabel='Amplitude')
    im = axes[1].imshow(x[0], origin='lower', aspect='auto', extent=[0, 5, 0, x.shape[1]])
    axes[1].set(title='Log-mel features', xlabel='Time (seconds)', ylabel='Mel band')
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(Path(output)/'audio_example.png', dpi=160)
    plt.close(fig)


def save_test_results(meta, truth, probabilities, classes, metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = probabilities.argmax(1)
    cm = np.zeros((50, 50), dtype=int)
    np.add.at(cm, (truth, pred), 1)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output/'confusion_matrix.csv')
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(50), where=support!=0)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(50), where=cm.sum(0)!=0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(50), where=precision+recall!=0)
    pd.DataFrame(dict(category=classes, precision=precision, recall=recall, f1=f1, support=support)).to_csv(output/'per_class_metrics.csv', index=False)
    rows = meta[meta.fold.isin(SPLITS['test'])].copy()
    rows['predicted_target'] = pred
    rows['predicted_category'] = [classes[i] for i in pred]
    rows['confidence'] = probabilities.max(1)
    rows['correct'] = pred==truth
    rows.to_csv(output/'predictions.csv', index=False)
    metrics['macro_f1'] = float(f1.mean())
    write_json(output/'test_metrics.json', metrics)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set(xticks=range(50), yticks=range(50), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True', title=f'ESC-50 fold 5: accuracy {metrics["accuracy"]:.1%}')
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax, label='Clips')
    fig.tight_layout()
    fig.savefig(output/'confusion_matrix.png', dpi=160)
    plt.close(fig)
