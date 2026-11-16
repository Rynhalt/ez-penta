"""Structural/export tests; no model execution or scientific validation is simulated."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data_ready.uk_extension import collect_jobs, export_dataset, make_jobs
from eval.dataset2_data import load_dataset, quality_flags
from eval.dataset2_analysis import observation_table, matched_observations, export_bundle


class UKExtensionTests(unittest.TestCase):
    def fixture(self):
        seed = dict(source_id='seed1', source_split='generated', split='tune', category='privacy',
                    subtopic='privacy', form='question', severity='clear', expected='decline',
                    mix_direction='frame_en+request_ja', prompts=28, complete=True)
        seeds = pd.DataFrame([seed])
        rows, texts, bases, techniques = [], [], [], []
        for kind in ['harmful', 'twin']:
            texts.append(dict(source_id='seed1', kind=kind, culture='neutral', ja='元の質問', en='Original question'))
            bases.append(dict(seed_id='seed1', kind=kind, strategy='base', ja='英国の質問', en='UK question', flags='provisional'))
            for strategy in ['rp', 'fc', 'rs']:
                techniques.append(dict(seed_id='seed1', kind=kind, strategy=strategy,
                                       ja='英国の質問です', en='UK framed question', mix='UK 質問', flags='provisional'))
            for framing in ['control', 'rp', 'fc', 'rs']:
                langs = [('ja','ja'), ('en','en')] + ([] if framing == 'control' else [('mix','ja'),('mix','en')])
                for lang, out in langs:
                    rows.append(dict(source_id='seed1', source_split='generated', split='tune', category='privacy',
                        subtopic='privacy', form='question', severity='clear', expected='answer' if kind == 'twin' else 'decline',
                        kind=kind, culture='neutral', framing=framing, prompt_language=lang, output_language=out,
                        mix_direction=seed['mix_direction'] if lang == 'mix' else None,
                        request='original request', prompt='unchanged original prompt', prompt_id=f'n-{kind}-{framing}-{lang}-{out}'))
        return pd.DataFrame(rows), seeds, pd.DataFrame(texts), bases, techniques

    def test_export_preserves_baseline_and_splits_and_matches_uk(self):
        neutral, seeds, texts, bases, techniques = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'dataset2.sqlite3'
            prompts, manifest = export_dataset(dest, neutral, seeds, texts, bases, techniques,
                                               dict(created_at='test', techniques=['rp','fc','rs']))
            loaded, ss = load_dataset(dest, expected_count=56, cultures=('neutral','uk'))
            self.assertTrue(ss.complete.all())
            self.assertEqual(set(loaded.split), {'tune'})
            pd.testing.assert_frame_equal(loaded[loaded.culture.eq('neutral')][neutral.columns].sort_values('prompt_id').reset_index(drop=True),
                                          neutral.sort_values('prompt_id').reset_index(drop=True), check_dtype=False)
            self.assertEqual(loaded.prompt_id.nunique(), 56)
            with self.assertRaisesRegex(ValueError, 'Invalid culture'):
                load_dataset(dest, expected_count=56)
            table = observation_table(quality_flags(loaded).assign(shard_id=1).to_dict('records'), {}, {})
            pairs = matched_observations(table)
            uk = pairs[pairs.comparison.eq('culture_vs_neutral')]
            self.assertEqual(len(uk), 56)  # 28 paired prompts, safety and refusal each
            self.assertEqual(set(uk.level_a), {'uk'})
            self.assertFalse(uk.known.any())
            self.assertFalse(table.expectation_validated.any())
            results, _ = export_bundle(Path(tmp)/'evaluation', table, {}, {},
                dict(expected_prompts_per_seed=56, prompt_ids=sorted(table.prompt_id)))
            self.assertIn('complete_seeds', set(results['sensitivity'].cohort))

    def test_invalid_and_truncated_json_never_exported_as_accepted(self):
        jobs = pd.DataFrame([dict(source_id='job',source_split='generated',seed_id='seed',kind='harmful',strategy='base')])
        for text, truncated in [('not JSON',False), ('{"ja":"日本語","en":"English"}',True),
                                ('{"ja":"日本語","en":""}',False)]:
            records = {('job','generated','paired'):dict(output=text,truncated=truncated,error=None)}
            accepted, audit = collect_jobs(jobs, records)
            self.assertEqual(accepted, [])
            self.assertFalse(audit.accepted.any())
        records = {('job','generated','paired'):dict(output='{"ja":"日本語","en":"English"}',truncated=False,error=None)}
        accepted, audit = collect_jobs(jobs, records)
        self.assertEqual(len(accepted),1)
        self.assertEqual(audit.review_status.iloc[0],'provisional')

    def test_job_identity_separates_twins_and_strategies(self):
        _, seeds, texts, _, _ = self.fixture()
        jobs = make_jobs(texts,seeds,stage='techniques',system='system',template='{ja} {en} {technique}',
                         culture_rule='UK',techniques={'rp':'role','fc':'fiction','rs':'rules'})
        self.assertEqual(len(jobs),6)
        self.assertEqual(jobs.source_id.nunique(),6)
        self.assertEqual(set(jobs.seed_id),{'seed1'})

    def test_changed_text_changes_new_id(self):
        neutral, seeds, texts, bases, techniques = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'dataset2.sqlite3'
            first,_ = export_dataset(dest,neutral,seeds,texts,bases,techniques,dict(created_at='test',techniques=[]))
            bases[0]['ja'] += '変更'
            second,_ = export_dataset(dest,neutral,seeds,texts,bases,techniques,dict(created_at='test',techniques=[]))
            self.assertNotEqual(set(first.prompt_id),set(second.prompt_id))
            self.assertEqual(set(first[first.culture.eq('neutral')].prompt_id),set(second[second.culture.eq('neutral')].prompt_id))


if __name__ == '__main__':
    unittest.main()
