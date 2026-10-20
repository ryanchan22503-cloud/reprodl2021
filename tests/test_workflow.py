"""Fast checks for split isolation, serialization, cache keys, and metrics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from esc50 import AudioNet, Features, feature_key, make_loader, save_test_results, setup


class WorkflowTests(unittest.TestCase):
    def test_folds_are_isolated(self):
        meta = pd.DataFrame({'fold': [1,2,3,4,5], 'target': [0,1,2,3,4]})
        features = torch.arange(5).reshape(5,1,1,1).float()
        observed = {}
        for split in ['train','validation','test']:
            observed[split] = {int(x) for batch, _ in make_loader(meta,features,split,2) for x in batch.flatten()}
        self.assertEqual(observed, {'train': {0,1,2}, 'validation': {3}, 'test': {4}})

    def test_checkpoint_roundtrip_preserves_predictions(self):
        setup(device='cpu')
        model = AudioNet(base_filters=4, dropout=.2).eval()
        x = torch.randn(2,1,64,157)
        expected = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'model.pt'
            torch.save({'state_dict': model.state_dict()},checkpoint)
            restored = AudioNet(base_filters=4, dropout=.2).eval()
            restored.load_state_dict(torch.load(checkpoint,weights_only=True)['state_dict'])
            torch.testing.assert_close(expected,restored(x))
            self.assertEqual(tuple(expected.shape),(2,50))

    def test_cache_changes_with_data_or_preprocessing(self):
        config = Features()
        key = feature_key('dataset-a',config)
        self.assertNotEqual(key,feature_key('dataset-b',config))
        self.assertNotEqual(key,feature_key('dataset-a',replace(config,n_mels=80)))
        self.assertNotEqual(key,feature_key('dataset-a',replace(config,normalize=False)))

    def test_metrics_handle_unpredicted_classes(self):
        names = [f'class{i}' for i in range(50)]
        truth = np.arange(50)
        probabilities = np.zeros((50,50))
        probabilities[:,0] = 1.
        meta = pd.DataFrame({'fold': [5]*50, 'target': truth, 'filename': [f'{i}.wav' for i in truth]})
        with tempfile.TemporaryDirectory() as tmp:
            save_test_results(meta,truth,probabilities,names,{'accuracy': .02},tmp)
            report = pd.read_csv(Path(tmp)/'per_class_metrics.csv')
            self.assertTrue(np.isfinite(report[['precision','recall','f1']]).all().all())
            self.assertAlmostEqual(report.iloc[0].precision,.02)
            self.assertEqual(report.iloc[1].recall,0.)
            cm = pd.read_csv(Path(tmp)/'confusion_matrix.csv',index_col=0)
            self.assertEqual(int(cm.to_numpy().sum()),50)
            self.assertIn('macro_f1',json.loads((Path(tmp)/'test_metrics.json').read_text()))


if __name__ == '__main__':
    unittest.main()
