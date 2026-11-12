import csv, json, random, sqlite3
from collections import defaultdict
from pathlib import Path
src=Path('/content/drive/MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/results_reduced.sqlite3')
out=Path('/content/drive/MyDrive/ez-penta/Pipeline1111/codex_extension_20261111/corrected_seed_only')
out.mkdir(parents=True, exist_ok=True)
c=sqlite3.connect(f'file:{src}?mode=ro', uri=True)
c.row_factory=sqlite3.Row
assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
labels={};
for line in (src.parent/'corrected_reanalysis'/'corrected_guard_labels.jsonl').open():
    z=json.loads(line); labels[z['prompt_id']]=z['guard_response_corrected']
q='''SELECT prompt_id,seed_id,split,kind,culture,prompt_language,analysis_eligible,generation_status,safety FROM observations'''
by=defaultdict(lambda: defaultdict(list)); counts=defaultdict(int); changed=0; recovered=0; published=0
for r in c.execute(q):
    s=labels.get(r['prompt_id'],'Missing');
    if s in ('Safe','Unsafe'): published+=1
    if r['split'] not in ('analysis','tune','heldout') or r['kind']!='harmful' or r['generation_status']!='completed' or str(r['analysis_eligible']).lower() not in ('1','true','yes') or r['prompt_language'] not in ('en','ja','mix') or s not in ('Safe','Unsafe'): continue
    counts[(r['split'],r['prompt_language'],r['culture'])]+=1
    by[(r['split'],r['prompt_language'])][(str(r['seed_id']),r['culture'])].append(1.0 if s=='Unsafe' else 0.0)
results=[]; seedrows=[]
for key, cells in sorted(by.items()):
    seeds=defaultdict(dict)
    for (seed,culture), vals in cells.items(): seeds[seed][culture]=sum(vals)/len(vals)
    effects=[(seed,vals['us']-vals['neutral']) for seed,vals in seeds.items() if 'us' in vals and 'neutral' in vals]
    rng=random.Random(20261111); vals=[e for _,e in effects]
    boots=[sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(5000)] if vals else []
    boots.sort()
    summary={'split':key[0],'prompt_language':key[1],'eligible_binary_harmful_rows':sum(counts[(key[0],key[1],x)] for x in ('us','neutral')),'matched_seeds':len(vals),'risk_difference':sum(vals)/len(vals) if vals else None,'ci95_seed_bootstrap':[boots[124],boots[4874]] if boots else None,'bootstrap_replicates':len(boots),'bootstrap_seed':20261111,'estimand':'equal-weight mean of within-seed unsafe-proxy mean(US)-mean(neutral); no within-seed framing/category cell match'}
    results.append(summary)
    for seed,effect in effects: seedrows.append({'split':key[0],'prompt_language':key[1],'seed_id':seed,'paired_difference':effect})
with (out/'seed_effects.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['split','prompt_language','seed_id','paired_difference']);w.writeheader();w.writerows(seedrows)
report={'source_sha256':'d6004f551eb8519f59ad090594df4373b5ade0e40756499675ee8deb4fb049df','source_rows':c.execute('select count(*) from observations').fetchone()[0],'label_basis':'corrected raw guard verdicts from full prior reparse; binary only','published_binary_labels':published,'comparisons':results,'limits':['Automated guard proxy, not human labels.','Associational seed-matched comparison, not causal.','No framing/category matching; pooled within each prompt-language group.','Heldout split previously exposed; descriptive only.']}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
c.close()
