"""Presentation-only plots from saved predictions and raw audio; never trains or tests a model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent


def create_visualizations(results_dir=None, data_dir=None):
    results_dir = Path(results_dir or ROOT/'artifacts/optimized')
    data_dir = Path(data_dir or ROOT/'data/ESC-50-master')
    output = results_dir/'visualizations'
    predictions = pd.read_csv(results_dir/'test/predictions.csv')
    metadata = pd.read_csv(data_dir/'meta/esc50.csv')
    expected = metadata[metadata.fold==5].set_index('filename')
    if predictions.filename.duplicated().any() or set(predictions.filename)!=set(expected.index):
        raise ValueError('Expected exactly the saved predictions for all held-out fold-5 recordings.')
    labels = expected.loc[predictions.filename]
    if not np.array_equal(predictions.target.to_numpy(),labels.target.to_numpy()) or not predictions.fold.eq(5).all():
        raise ValueError('Saved prediction labels do not match the held-out dataset.')
    classes = metadata.drop_duplicates('target').set_index('target').category
    if not np.array_equal(predictions.category, predictions.target.map(classes)) or not np.array_equal(predictions.predicted_category,predictions.predicted_target.map(classes)):
        raise ValueError('Prediction class names do not match their target numbers.')
    if not np.isfinite(predictions.confidence).all() or not predictions.confidence.between(0,1).all():
        raise ValueError('Confidence must be finite and between zero and one.')
    predictions['correct'] = predictions.target.eq(predictions.predicted_target)
    errors = predictions[~predictions.correct]
    correct = predictions[predictions.correct]
    if errors.empty or correct.empty:
        raise ValueError('These comparison plots require both correct and incorrect predictions.')
    output.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        # Directed mistakes: true category -> predicted category, excluding the diagonal.
        pairs = errors.groupby(['category','predicted_category']).size().reset_index(name='count')
        pairs = pairs.sort_values(['count','category','predicted_category'],ascending=[False,True,True]).head(10)
        fig,ax=plt.subplots(figsize=(11,5.5))
        labels=[f'{a.replace("_"," ")}  →  {b.replace("_"," ")}' for a,b in zip(pairs.category,pairs.predicted_category)]
        bars=ax.barh(labels,pairs['count'],color='#dc7848')
        ax.invert_yaxis()
        ax.bar_label(bars,padding=5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0,float(pairs['count'].max())+1)
        ax.set(xlabel='Misclassified test recordings (count)',title='Most confused sound pairs: true → predicted')
        fig.text(.5,.015,f'Top {len(pairs)} directed pairs among {len(errors)} mistakes. Each category has 8 test recordings; ties use alphabetical order.',ha='center',fontsize=9,color='#555555')
        fig.tight_layout(rect=[0,.04,1,1])
        fig.savefig(output/'most_confused_pairs.png',dpi=160)
        plt.close(fig)
        pairs.to_csv(output/'most_confused_pairs.csv',index=False)

        fig,ax=plt.subplots(figsize=(10.5,4.7))
        for group,label,color in [(correct,'Correct','#16856b'),(errors,'Incorrect','#dc7848')]:
            values=group.confidence.to_numpy()*100
            ax.hist(values,bins=np.linspace(0,100,11),weights=np.full(len(values),100/len(values)),
                    alpha=.45,label=f'{label} ({len(values)} recordings)',color=color,edgecolor=color)
            ax.axvline(np.median(values),color=color,linestyle='--',linewidth=1.5)
        ax.set(xlim=(0,100),xlabel='Model confidence (uncalibrated)',ylabel='Share within each outcome (%)',
               title='Confidence when the model is right or wrong')
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.legend()
        fig.text(.5,.015,'Each group is normalized separately. Dashed lines show median confidence; confidence is not a guaranteed probability of being right.',ha='center',fontsize=8.5,color='#555555')
        fig.tight_layout(rect=[0,.05,1,1])
        fig.savefig(output/'confidence_correct_vs_incorrect.png',dpi=160)
        plt.close(fig)

        # Illustrations: first alphabetical mistake in the top confusion pair, and a correct clip
        # of the same true category where available. Never modify labels or predictions.
        chosen_pair = pairs.iloc[0]
        wrong = errors[(errors.category==chosen_pair.category)&(errors.predicted_category==chosen_pair.predicted_category)].sort_values('filename').iloc[0]
        matching = correct[correct.category==wrong.category].sort_values('filename')
        right = matching.iloc[0] if len(matching) else correct.sort_values('filename').iloc[0]
        config=json.loads((results_dir/'run_config.json').read_text())['features']
        torch.set_num_threads(4)
        mel=torchaudio.transforms.MelSpectrogram(sample_rate=config['sample_rate'],n_fft=config['n_fft'],
                                                hop_length=config['hop_length'],n_mels=config['n_mels'])
        db=torchaudio.transforms.AmplitudeToDB(top_db=config['top_db'])
        examples=[]
        for row in [right,wrong]:
            wav,sr=sf.read(data_dir/'audio'/row.filename,dtype='float32',always_2d=True)
            mono=wav.mean(1)
            tensor=torch.from_numpy(mono).unsqueeze(0)
            if sr!=config['sample_rate']:
                tensor=torchaudio.transforms.Resample(sr,config['sample_rate'])(tensor)
            length=config['sample_rate']*5
            tensor=torch.nn.functional.pad(tensor[...,:length],(0,max(0,length-tensor.shape[-1])))
            spec=db(mel(tensor))
            if config['normalize']:
                spec=(spec-spec.mean())/spec.std().clamp_min(1e-6)
            examples.append((row,mono,sr,spec[0].numpy()))
        vmin=min(example[3].min() for example in examples)
        vmax=max(example[3].max() for example in examples)
        fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        for index,(row,wav,sr,spec) in enumerate(examples):
            outcome='Correct' if row.correct else 'Incorrect'
            color='#16856b' if row.correct else '#dc7848'
            axes[index,0].plot(np.arange(len(wav))/sr,wav,color=color,linewidth=.5)
            axes[index,0].set(xlabel='Time (seconds)',ylabel='Amplitude',xlim=(0,5),
                              title=f'{outcome}: true {row.category.replace("_"," ")} | predicted {row.predicted_category.replace("_"," ")}')
            axes[index,0].text(.01,.03,f'{row.filename}  |  confidence {row.confidence:.1%}',transform=axes[index,0].transAxes,fontsize=8)
            image=axes[index,1].imshow(spec,origin='lower',aspect='auto',extent=[0,5,0,config['n_mels']],
                                       cmap='magma',vmin=vmin,vmax=vmax)
            axes[index,1].set(xlabel='Time (seconds)',ylabel='Mel band',title=f'{outcome} example: model input spectrogram')
        fig.colorbar(image,ax=axes[:,1],label='Standardized log-mel energy' if config['normalize'] else 'Log-mel energy (dB)',shrink=.85)
        fig.suptitle('Two illustrative recordings: waveform and corresponding audio features',fontsize=13)
        fig.savefig(output/'waveform_spectrogram_examples.png',dpi=160)
        plt.close(fig)
    details=dict(samples=len(predictions),correct=len(correct),incorrect=len(errors),
                 accuracy=float(predictions.correct.mean()),
                 median_confidence_correct=float(correct.confidence.median()),
                 median_confidence_incorrect=float(errors.confidence.median()),
                 correct_example=right.filename,incorrect_example=wrong.filename,
                 example_selection='Alphabetically first mistake in the most frequent directed confusion pair; first correct clip of the same true category if available, otherwise first correct clip overall.',
                 note='Illustrative examples, not a random or representative sample. No model predictions, weights, or scores are changed.')
    (output/'visualization_summary.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    return output


def display_confusion_matrix(path):
    """Embed PNG bytes so the disclosure remains usable in a saved notebook."""
    import base64
    from IPython.display import HTML, display
    encoded=base64.b64encode(Path(path).read_bytes()).decode('ascii')
    display(HTML('<details><summary style="cursor:pointer;font-weight:bold;padding:12px 0">'
                 'Full confusion matrix — click to expand (50 sound categories)</summary>'
                 '<p>Rows are true categories; columns are predictions. Diagonal cells are correct answers.</p>'
                 f'<img alt="Full 50-category confusion matrix" style="max-width:100%;height:auto" src="data:image/png;base64,{encoded}">'
                 '</details>'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'artifacts/optimized')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/ESC-50-master')
    args=parser.parse_args()
    print(create_visualizations(args.results_dir,args.data_dir))
