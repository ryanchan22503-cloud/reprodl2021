"""Compare paired predictions from the original-settings and standardized runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT/'outputs/comparison'
    names = ['Original settings', 'Updated settings']
    paths = [output/'original', output/'standardized']
    metrics = [json.loads((p/'test/test_metrics.json').read_text()) for p in paths]
    predictions = [pd.read_csv(p/'test/predictions.csv') for p in paths]
    histories = [pd.read_csv(p/'history.csv') for p in paths]
    pd.testing.assert_frame_equal(predictions[0][['filename','target','fold']], predictions[1][['filename','target','fold']])
    a, b = [p.correct.to_numpy(dtype=bool) for p in predictions]
    paired = dict(both_correct=int((a&b).sum()), updated_only_correct=int((~a&b).sum()),
                  original_only_correct=int((a&~b).sum()), both_wrong=int((~a&~b).sum()))
    report = {'original': metrics[0], 'updated': metrics[1], 'paired': paired,
              'accuracy_gain_percentage_points': 100*(metrics[1]['accuracy']-metrics[0]['accuracy']),
              'additional_correct_clips': int(b.sum()-a.sum()),
              'note': 'Single seed and held-out fold; original is a compatibility adaptation; updated code remains local.'}
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(1,3,figsize=(15,4.5))
    colors = ['#52758c','#16856b']
    for label, history, color in zip(names,histories,colors):
        axes[0].plot(history.epoch,history.val_accuracy*100,label=label,color=color)
        axes[1].plot(history.epoch,history.val_loss,label=label,color=color)
    axes[0].set(title='Validation accuracy',xlabel='Training epoch',ylabel='Accuracy (%)',ylim=(0,100))
    axes[1].set(title='Validation loss',xlabel='Training epoch',ylabel='Cross-entropy loss')
    axes[0].legend()
    values = [100*m['accuracy'] for m in metrics]
    bars = axes[2].bar(names,values,color=colors,width=.55)
    axes[2].bar_label(bars,labels=[f'{v:.2f}%' for v in values],padding=4)
    axes[2].set(title='Held-out test accuracy (400 clips)',ylabel='Accuracy (%)',ylim=(0,100))
    fig.suptitle('ESC-50: original vs updated settings | seed 0, 25 epochs, same folds')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
