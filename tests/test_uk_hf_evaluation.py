"""CPU checks for adapter provenance and paired UK comparisons; no GPU claims."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import pandas as pd
from eval.uk_hf_evaluation import find_adapter, adapter_identity, paired_changes

class UKHFEvaluationTests(unittest.TestCase):
    def test_reference_guard_token_cap_reaches_both_assessments(self):
        from eval.dataset2_runtime import make_judge
        class Runner:
            configuration = {'model_id': 'fixture-guard'}
            calls = []
            def generate(self, messages, max_new_tokens):
                self.calls.append(max_new_tokens)
                return dict(status='completed', answer='Safety: Safe\nCategories: None\nRefusal: No')
        runner = Runner()
        score, _ = make_judge(runner, max_tokens=64)
        result = score({'prompt_id':'test','prompt':'fixture'}, {'answer':'fixture','identity':'test'})
        self.assertEqual(runner.calls,[64,64])
        self.assertEqual(result['status'],'completed')

    def test_adapter_lookup_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with zipfile.ZipFile(root/'pipeline4_adapters.zip','w') as z:
                z.writestr('pipeline4_restricted_adapter/adapter_config.json','{}')
                z.writestr('pipeline4_restricted_adapter/adapter_model.safetensors',b'fixture')
            adapter=find_adapter(root,'restricted',local_root=root/'local')
            hashes=adapter_identity(adapter)
            self.assertEqual(set(hashes),{'adapter_config.json','adapter_model.safetensors'})
            with self.assertRaises(FileNotFoundError):
                find_adapter(root,'broad',local_root=root/'local')

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with zipfile.ZipFile(root/'pipeline4_adapters.zip','w') as z:
                z.writestr('pipeline4_restricted_adapter/../../outside','bad')
            with self.assertRaises(ValueError):
                find_adapter(root,'restricted',local_root=root/'local')

    def tables(self):
        base=pd.DataFrame([dict(prompt_id=str(i),prompt='same',seed_id='seed'+str(i//2),split='tune',
            culture='uk',prompt_language='ja',kind='harmful',unsafe=1.,refused=0.) for i in range(4)])
        fixed=base.copy();fixed['unsafe']=0.;fixed['refused']=1.
        return base,fixed

    def test_paired_metric_and_seed_units(self):
        base,fixed=self.tables()
        fixed.loc[0,'unsafe']=float('nan')
        summary=paired_changes({'baseline':base,'restricted':fixed})
        unsafe=summary[summary.metric.eq('unsafe')].iloc[0]
        self.assertEqual(unsafe.known_pairs,3)
        self.assertEqual(unsafe.known_seeds,2)
        self.assertEqual(unsafe.one_to_zero,3)
        self.assertEqual(unsafe.seed_mean_delta,-1.)

    def test_different_prompt_inventory_rejected(self):
        base,fixed=self.tables()
        with self.assertRaises(ValueError):
            paired_changes({'baseline':base,'restricted':fixed.iloc[:-1]})
        fixed.loc[0,'prompt']='different'
        with self.assertRaises(ValueError):
            paired_changes({'baseline':base,'restricted':fixed})
