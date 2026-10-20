# Original vs updated settings — 20 October 2026

## What was compared

GitHub's `ryanchan22503-cloud/reprodl2021` branches `main` and `test-branch` were identical at
`b391c31576518eb6a4383e1b045da59d34ebbc97` when checked through the GitHub connector.
The notebook blob on the fork and original upstream repository was identical:
`aaabb683d6877753eb895d9b03b9109bb328a06c`.
The updated workflow files remain local and uncommitted; no GitHub code was published for this comparison.

Training ran locally on the laptop's RTX 4060, not on GitHub-hosted hardware.
The original notebook's import check failed in the project's modern environment because `pytorch_lightning`
is not installed. It also has historical Lightning APIs and an unfinished test loop.
The accuracy comparison therefore uses a compatibility adaptation of its settings through the supported shared trainer.
The original architecture's output was verified to match the shared AudioNet exactly after loading identical weights.
The original mel defaults were checked against `torchaudio.transforms.MelSpectrogram(sample_rate=8000)`.
Evidence and original notebook snapshot are in the ignored `outputs/comparison/` folder.

## Common conditions

Both models were trained afresh for 25 epochs with seed 0, the same four-convolution AudioNet,
Adam at learning rate 0.001, no dropout or weight decay, and no learning-rate scheduler.
Both used ESC-50 folds 1–3 (1,200 clips) for training, fold 4 (400) for selecting the best checkpoint,
and fold 5 (400) for testing. Each saved model was tested in a new process.
Both runs used Python 3.12.11, PyTorch/torchaudio 2.8.0+cu129, and NumPy 2.2.4.

| Setting | Original compatibility version | Updated version |
| --- | --- | --- |
| Audio sample rate | 8 kHz | 8 kHz |
| Mel bands | 128 | 64 |
| FFT size / hop | 400 / 200 | 512 / 256 |
| Per-clip standardization | Off | On |
| Batch size | 8 | 32 |
| Epochs | 25 | 25 |
| Best validation checkpoint | Epoch 25 | Epoch 24 |

## Results

| Metric | Original settings | Updated settings |
| --- | --- | --- |
| Best validation accuracy | 48.75% | 60.25% |
| Held-out test accuracy | 45.25% | 54.00% |
| Correct test clips | 181 / 400 | 216 / 400 |
| Test cross-entropy (lower is better) | 1.973435 | 1.682542 |
| Test macro F1 | 0.423619 | 0.510740 |

The updated setup gained **8.75 percentage points**, or **35 additional correct predictions**.
On individual recordings, both were correct on 149 clips; the updated setup alone was correct on 67,
the original setup alone on 32, and both were wrong on 152.

These are single-seed, single-test-fold results. Multiple preprocessing settings and batch size changed together;
the experiment does not isolate the effect of any one change. Equal epochs also mean different optimizer-step
counts because batch sizes differ. No settings were adjusted based on these test outcomes.
The repeated updated run reproduced the earlier 54% result; this does not establish multi-seed reliability.

## Reproduce

Run from the repository folder, choosing new output folders if the ones below already contain results:

```powershell
.\.venv\Scripts\python train.py --epochs 25 --feature-preset original --batch-size 8 --output-dir outputs/comparison/original
.\.venv\Scripts\python test.py --checkpoint outputs/comparison/original/audionet.ckpt --batch-size 8
.\.venv\Scripts\python train.py --epochs 25 --output-dir outputs/comparison/standardized
.\.venv\Scripts\python test.py --checkpoint outputs/comparison/standardized/audionet.ckpt
.\.venv\Scripts\python compare_results.py
```

`compare_results.py` reads those two run folders and writes `outputs/comparison/comparison.json` and `comparison.png`.
Each model folder contains its checkpoint, settings, history, plots, and detailed test predictions.
Four workflow tests passed after adding the original feature preset.
