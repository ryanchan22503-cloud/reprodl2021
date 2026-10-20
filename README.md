# ESC-50 audio classification

Train an audio classifier, save its best validation model, and evaluate it on held-out audio.
This branch adapts Simone Scardapane's MIT-licensed [Reproducible Deep Learning course](https://github.com/sscardapane/reprodl2021).
The original `Initial Notebook.ipynb`, license, and [upstream instructions](docs/UPSTREAM_README.md) are preserved.
Use the new scripts or notebooks below; the original notebook retains its historical APIs.

## Environment

Use **Python 3.12** and run these commands from the repository folder.
A standalone `.venv` has already been prepared on this laptop; it does not reference the ComfyUI environment.
For a fresh installation on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m ipykernel install --user --name reprodl2021 --display-name "ESC-50 (repo .venv)"
```

`requirements.txt` pins the direct project dependencies. For the exact validated Windows/Python 3.12
dependency versions, install `requirements-lock.txt` instead. Both files include the GPU wheel index.

The pinned PyTorch wheels target Windows/Linux with NVIDIA CUDA 12.9 support. They work on this laptop's verified GPU setup;
other machines may need a different wheel build. For CPU-only installation, replace the first two requirements
with `torch==2.8.0` and `torchaudio==2.8.0`, and use the official CPU wheel index when installing those packages.
macOS requires its platform-supported PyTorch wheels. Runtime selection defaults to GPU when available and otherwise CPU.

PowerShell does not need environment activation when using the explicit interpreter paths above.
In Git Bash, the equivalent interpreter path is `./.venv/Scripts/python.exe`.

## Dataset

Download and extract [ESC-50](https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip) into `data/`:

```text
data/ESC-50-master/
    audio/          # 2,000 WAV files
    meta/esc50.csv
```

The existing dataset has been copied into this repository's ignored `data/` folder on this laptop.
For another location, pass `--data-dir "path/to/ESC-50-master"` to both scripts.
Metadata, class counts, filenames, and the presence of all audio files are checked before training.
Dataset licensing and attribution are separate from the code license; retain the dataset's README and license files.

## Train and test

```powershell
.\.venv\Scripts\python train.py --epochs 25
.\.venv\Scripts\python test.py
```

Alternatively open `train.ipynb`, select **ESC-50 (repo .venv)**, and run all cells. Then run `test.ipynb` with the same kernel.
The notebooks should run from the repository directory. Their data/output settings are in the first code cell.
The notebooks and scripts use the same shared model and preprocessing functions.

| Split | Folds | Clips | Purpose |
| --- | --- | --- | --- |
| Training | 1, 2, 3 | 1,200 | Update model weights |
| Validation | 4 | 400 | Select the best epoch |
| Test | 5 | 400 | Evaluate the selected checkpoint |

The train script never evaluates fold 5. The test script reconstructs the model and feature settings from the checkpoint,
checks class names and the metadata fingerprint, and evaluates only fold 5 in inference mode.
Feature caching may process all clips, but it fits no statistics across clips; normalization is independent for every clip.

Training preserves the original four-convolution AudioNet architecture, with Adam at learning rate 0.001 and no dropout,
weight decay, or scheduler by default. The runnable baseline uses explicit 8 kHz, 64-band log-mel features
(`n_fft=512`, `hop_length=256`, `top_db=80`) and per-clip standardization.
These preprocessing settings differ from the original notebook's implicit defaults, so results are not an exact reproduction.
Audio uses its actual sample rate, is mixed to mono, and is padded or truncated to five seconds.
Feature caches invalidate when settings, metadata, audio size/timestamps, or PyTorch/torchaudio versions change.

Each training run needs an empty output directory to prevent accidentally replacing a previous model.
The default is `outputs/baseline`. For a new experiment:

```powershell
.\.venv\Scripts\python train.py --epochs 25 --dropout 0.3 --weight-decay 0.001 --scheduler cosine --output-dir outputs/experiment-01
.\.venv\Scripts\python test.py --checkpoint outputs/experiment-01/audionet.ckpt
```

Choose optimization settings using validation results before evaluating the final candidate on test data.
These options are available for experiments; improved accuracy is not guaranteed.
Use `--device cpu` to force CPU, `--device cuda` to require GPU, and `--seed`, `--batch-size`,
`--learning-rate`, and `--cache-dir` as needed. See each script's `--help` for all options.
Seeded runs improve repeatability; exact results across GPU hardware/software are not guaranteed.

To compare with the original notebook's feature settings and batch size using the same supported training code:

```powershell
.\.venv\Scripts\python train.py --epochs 25 --feature-preset original --batch-size 8 --output-dir outputs/comparison/original
.\.venv\Scripts\python test.py --checkpoint outputs/comparison/original/audionet.ckpt --batch-size 8
```

The `original` preset uses 128 mel bands, FFT size 400, hop length 200, and no per-clip standardization.
This is a compatibility adaptation: the historical Lightning APIs and unfinished test loop are replaced.
Both presets use the best validation checkpoint for a consistent comparison.

## Outputs

Training writes to `outputs/baseline/` by default:

- `audionet.ckpt`: best validation model, class names, feature settings, split definition, and run configuration.
- `history.csv` and `training_curves.png`: epoch losses, accuracies, and learning rates.
- `audio_example.png`: waveform and corresponding log-mel features.
- `run_config.json`: settings, environment, and dataset fingerprints.
- `train_metrics.json`: selected epoch and validation accuracy.

Testing writes to the checkpoint directory's `test/` subfolder:

- `test_metrics.json`: accuracy, loss, macro F1, selected epoch, and checkpoint fingerprint.
- `confusion_matrix.png` / `.csv`: errors across the 50 classes.
- `per_class_metrics.csv`: precision, recall, F1, and support; undefined precision is recorded as zero.
- `predictions.csv`: each clip's true/predicted category, softmax confidence, and correctness.

Softmax confidence is not calibrated. Checkpoint files use this branch's PyTorch format and are not interchangeable
with original Lightning checkpoints or the earlier `20-10` three-block model.
A single test fold is an introductory baseline, not the official five-fold mean. Random guessing has expected accuracy 2%.

## Verification

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m pip check
```

Checks cover fold isolation, checkpoint prediction round trips, feature cache invalidation keys, and metrics for classes
with no predicted examples. Full dataset train/test results are recorded in [docs/VALIDATION.md](docs/VALIDATION.md).
Virtual environments, datasets, caches, and generated models/results are ignored by Git.
