# newcode1: changes from the first iteration

`newcode1` names the optimized 67% version. Its executed notebook is [newcode1.ipynb](../newcode1.ipynb).
The saved optimized model is `artifacts/optimized/audionet.ckpt`.

There are two earlier reference points: the original course notebook and our first working repository version.
The original notebook had historical Lightning APIs and an unfinished test loop; its reported score below
comes from a compatibility adaptation of its architecture/settings, not the untouched notebook.
Our earliest separate `20-10` experiment used a different three-block model and scored 45%; it is not the
first repository baseline used in the 54%-to-67% comparison.

| Change | Original course settings, compatibility run | First working repository iteration | newcode1 |
| --- | --- | --- | --- |
| Test accuracy | 45.25% | 54.00% | **67.00%** |
| Correct recordings | 181 / 400 | 216 / 400 | **268 / 400** |
| Best validation accuracy | 48.75% | 60.25% | **81.00%** |
| Audio sample rate | 8 kHz | 8 kHz | **16 kHz** |
| Mel bands | 128 | 64 | 64 |
| FFT / hop size | 400 / 200 | 512 / 256 | **1,024 / 320** |
| Per-recording normalization | No | Yes | Yes |
| Convolution layers | 4 | 4 | **8** |
| Pooling | Global average | Global average | **Time mean/max, retaining four frequency regions** |
| Training-only augmentation | None | None | **Time shifts and time/frequency masking** |
| Dropout | None | None | **0.1 in convolution blocks, 0.4 before classifier** |
| Optimizer | Adam | Adam | **AdamW**, weight decay 0.0001 |
| Learning-rate schedule | Fixed | Fixed | **Cosine decay** |
| Label smoothing | None | None | **0.05** |
| Batch size | 8 | 32 | 32 |
| Training epochs | 25 | 25 | **80** |
| Chosen epoch | 25 | 24 | **69** |

In plain language, newcode1 retains more sound detail, learns more complex patterns, practices with altered
training examples, and trains longer. Regularization reduces reliance on particular training examples.
The individual contribution of each change was not isolated, so the accuracy gain cannot be attributed to one technique.

## Workflow improvements over the original notebook

- Separate train and test scripts and reusable model/data code.
- A standalone environment, pinned dependencies, configurable paths, and automatic GPU/CPU selection.
- Dataset checks and cached audio features.
- Saving/loading the best validation checkpoint with model settings and class labels.
- Training plots, audio plots, confusion matrices, per-class scores and individual predictions.
- Six workflow checks, including new-architecture reconstruction and augmentation cache integrity.
- A reproducible two-candidate search in `optimize.py`, plus a published trained model and executed notebook.

Most basic workflow improvements were already present in the first working repository version. The richer features,
eight-layer model, augmentation, regularization, longer training, and two-candidate search are the further optimization.
Mixup was tested in candidate B but **is not used by the selected newcode1 model**.

## How the comparison was kept separate from testing

All versions use folds 1–3 for training, fold 4 for model selection, and fold 5 for final testing.
The further search selected candidate A at 81% validation accuracy over candidate B at 80.25%.
Only that selected candidate was evaluated on fold 5; no settings were changed after its test result.
The result is **13 percentage points** above the first repository iteration, or **52 additional correct clips**.
Against the original-settings compatibility run, it is 21.75 points higher, or 87 additional correct clips.
These are single-seed, single-fold results with different training durations, not an official five-fold benchmark.

## Open or test newcode1

Open `newcode1.ipynb` in Jupyter with the repository Python environment. Its saved cell outputs are also
viewable directly on GitHub. The notebook loads and tests the included model; it does not retrain for 80 epochs.

From Git Bash, after environment and dataset setup:

```bash
./.venv/Scripts/python.exe test.py --checkpoint artifacts/optimized/audionet.ckpt --output-dir outputs/newcode1-test
```

Use the explicit checkpoint to test newcode1: the default `train.py` and `test.py` commands still target the earlier
baseline for backward compatibility. See [OPTIMIZATION.md](OPTIMIZATION.md) for full training commands.
