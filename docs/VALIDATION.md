# Validation: 20 October 2026

Validated locally on `test-branch` using a standalone repository `.venv` and the complete ESC-50 dataset.
The environment was populated by copying the required installed distributions into this new environment;
no `.pth` reference to the older project or ComfyUI environment is used. `requirements-lock.txt` records its versions.

## Environment and checks

- Windows, Python 3.12.11.
- PyTorch and torchaudio 2.8.0+cu129, NumPy 2.2.4.
- NVIDIA GeForce RTX 4060 Laptop GPU.
- `python -m pip check`: no broken requirements.
- Four workflow tests passed: fold isolation, checkpoint prediction round trip, cache-key invalidation, and zero-prediction class metrics.
- Both new notebook files passed notebook-format validation. Training and evaluation were executed through the shared command-line scripts.
- Original notebook and MIT license were not modified. Dataset, environment, caches, and outputs are ignored by Git.

## Complete baseline run

```powershell
.\.venv\Scripts\python train.py --epochs 25
.\.venv\Scripts\python test.py
```

| Measurement | Result |
| --- | --- |
| Training clips / epochs | 1,200 / 25 |
| Validation clips | 400 (fold 4) |
| Selected epoch | 24 |
| Best validation accuracy | 60.25% |
| Test clips | 400 (fold 5) |
| Test correct predictions | 216 / 400 |
| Test accuracy | 54.00% |
| Test cross-entropy loss | 1.682542 |
| Test macro F1 | 0.510740 |

The test was run in a fresh Python process using the saved best checkpoint.
The baseline uses the original model architecture and the explicitly documented new preprocessing.
No architecture or hyperparameter change was made in response to the test result.
This is a single-seed, single-test-fold result; no official five-fold mean or controlled optimization comparison is claimed.
The earlier `20-10` model used a different architecture and training setup, so its 45% test result is not a controlled comparison.

Local model: `outputs/baseline/audionet.ckpt`.
Training plots and history are in `outputs/baseline/`; test figures, reports, and predictions are in `outputs/baseline/test/`.
To train again without overwriting this run, choose a new `--output-dir`.
