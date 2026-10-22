"""Run a predefined two-candidate validation search; optionally test only its winner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from esc50 import ROOT, write_json

COMMON = dict(architecture='spectrogram', feature_preset='highres', epochs=80, batch_size=32,
              seed=0, learning_rate=.001, weight_decay=.0001, optimizer='adamw', scheduler='cosine',
              augment=True, label_smoothing=.05)
CANDIDATES = {'candidate-a':dict(COMMON,dropout=.4,mixup=0.),
              'candidate-b':dict(COMMON,dropout=.3,mixup=.2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/optimization')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/ESC-50-master')
    parser.add_argument('--reuse-completed',action='store_true',help='Reuse only completed runs with matching settings')
    parser.add_argument('--evaluate-test',action='store_true',help='After selection, evaluate only the winner on fold 5')
    args = parser.parse_args()
    records = []
    for name,config in CANDIDATES.items():
        folder = args.output_dir/name
        complete = folder/'train_metrics.json'
        if complete.exists() and args.reuse_completed:
            saved = json.loads((folder/'run_config.json').read_text())
            if any(saved.get(key)!=value for key,value in config.items()):
                raise ValueError(f'Existing run settings differ: {folder}. Choose a new output directory.')
            if Path(saved['data_dir']).resolve()!=args.data_dir.resolve():
                raise ValueError(f'Dataset path differs for {folder}')
            print(f'Reusing completed validation run: {name}',flush=True)
        else:
            command = [sys.executable,str(ROOT/'train.py'),'--data-dir',str(args.data_dir),
                       '--output-dir',str(folder)]
            for key,value in config.items():
                command.append('--'+key.replace('_','-'))
                if not isinstance(value,bool):
                    command.append(str(value))
            subprocess.run(command,cwd=ROOT,check=True)
        result = json.loads(complete.read_text())
        if result['epochs']!=config['epochs'] or not (folder/'audionet.ckpt').is_file():
            raise ValueError(f'Incomplete run: {folder}')
        records.append(dict(candidate=name,validation_accuracy=result['best_validation_accuracy'],
                            selected_epoch=result['selected_epoch'],checkpoint=str((folder/'audionet.ckpt').resolve()),
                            settings=config))
    # Candidate order breaks exact accuracy ties. Test results are never read by this script.
    winner = max(records,key=lambda record:record['validation_accuracy'])
    report = dict(selection_metric='fold 4 validation accuracy',test_used_for_selection=False,
                  candidates=records,winner=winner['candidate'],checkpoint=winner['checkpoint'])
    write_json(args.output_dir/'selection.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if args.evaluate_test:
        subprocess.run([sys.executable,str(ROOT/'test.py'),'--data-dir',str(args.data_dir),
                        '--checkpoint',winner['checkpoint']],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
