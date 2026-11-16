import sqlite3, json, re, math, collections, sys
import pandas as pd
from scipy.stats import binomtest

import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[2])
c = sqlite3.connect(f'{ROOT}/notebooks/combined/results.sqlite3')
df = pd.read_sql("""select prompt_id, seed_id, split, category, subtopic, form, severity, kind, expected,
  culture, framing, prompt_language, mix_direction, output_language, safety, refused, raw_guard_outputs,
  generation_status, scoring_status, analysis_eligible from observations""", c)

def outcat(raw):
    if not raw: return None
    try:
        a = json.loads(raw)[-1]['answer']
    except Exception:
        return None
    m = re.search(r'Categories:\s*(.*)', a)
    return m.group(1).strip() if m else None
df['out_cat'] = df.raw_guard_outputs.map(outcat)
df = df.drop(columns='raw_guard_outputs')
df['known'] = df.safety.isin(['Safe', 'Unsafe'])
df['unsafe'] = (df.safety == 'Unsafe').astype(int)
df['ref'] = pd.to_numeric(df.refused, errors='coerce')
df['lang'] = df.prompt_language
nh = df[df.split != 'heldout']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 400)
out = {}

def rate_table(d, rowv, colv, val):
    g = d.groupby([rowv, colv])[val].agg(['sum', 'count'])
    r = (g['sum'] / g['count'] * 100).round(1).unstack()
    n = g['count'].unstack()
    return r, n

H = nh[(nh.kind == 'harmful') & nh.known]
T = nh[(nh.kind == 'twin') & nh.ref.notna()]
B = nh[(nh.kind == 'harmful') & (nh.expected != 'decline') & nh.ref.notna()]
print('expected values', nh.expected.unique(), 'severity', nh.severity.unique())
print('\n### Missingness (no Safe/Unsafe label) by category, harmful, non-heldout')
hh = nh[nh.kind == 'harmful']
print((1 - hh.groupby('category').known.mean()).round(3).sort_values())
print('missing by framing', (1 - hh.groupby('framing').known.mean()).round(3).to_dict())
print('missing by lang', (1 - hh.groupby('lang').known.mean()).round(3).to_dict())

for dim in ['framing', 'lang', 'culture', 'severity', 'form']:
    for name, d, val in [('UNSAFE% harmful', H, 'unsafe'), ('OVER-REFUSAL% twins', T, 'ref')]:
        r, n = rate_table(d, 'category', dim, val)
        print(f'\n### GGUF baseline {name}: category x {dim} (rate %, then n)')
        print(r.to_string()); print(n.to_string())

print('\n### Over-caution% (refusal on harmful expected!=decline) by category x framing')
r, n = rate_table(B, 'category', 'framing', 'ref'); print(r.to_string()); print(n.to_string())

print('\n### Output-category (Qwen3Guard) of UNSAFE answers: prompt category x output category (counts)')
U = H[H.unsafe == 1]
ct = pd.crosstab(U.category, U.out_cat, margins=True)
print(ct.to_string())
print('\nrow-normalised %'); print((pd.crosstab(U.category, U.out_cat, normalize='index') * 100).round(0).to_string())
print('\nOutput category x framing (unsafe answers)'); print(pd.crosstab(U.out_cat, U.framing, margins=True).to_string())
print('\nOutput category x language'); print(pd.crosstab(U.out_cat, U.lang, margins=True).to_string())
print('\nOutput category x culture'); print(pd.crosstab(U.out_cat, U.culture, margins=True).to_string())
print('\nUnsafe answers on TWINS (harmless prompts) by category / out_cat')
TU = nh[(nh.kind == 'twin') & (nh.safety == 'Unsafe')]
print(pd.crosstab(TU.category, TU.out_cat, margins=True).to_string())

# ---- paired within-seed contrasts per category (exact McNemar on matched pairs)
def paired(d, dim, a, b, keys, val):
    A = d[d[dim] == a].groupby(keys)[val].max(); Bv = d[d[dim] == b].groupby(keys)[val].max()
    j = pd.concat([A.rename('a'), Bv.rename('b')], axis=1, join='inner').dropna()
    up = int(((j.a == 1) & (j.b == 0)).sum()); dn = int(((j.a == 0) & (j.b == 1)).sum())
    p = binomtest(up, up + dn, 0.5).pvalue if up + dn else 1.0
    return len(j), up, dn, p, j.a.mean() * 100, j.b.mean() * 100

contrasts = [
    ('framing', 'rp', 'control', ['seed_id', 'culture', 'lang', 'output_language']),
    ('framing', 'fc', 'control', ['seed_id', 'culture', 'lang', 'output_language']),
    ('framing', 'rs', 'control', ['seed_id', 'culture', 'lang', 'output_language']),
    ('culture', 'us', 'neutral', ['seed_id', 'framing', 'lang', 'output_language']),
    ('culture', 'japan', 'neutral', ['seed_id', 'framing', 'lang', 'output_language']),
    ('lang', 'ja', 'en', ['seed_id', 'framing', 'culture']),
]
Hk = H.copy(); Tk = T.copy()
rows = []
for metric, d, val in [('unsafe', Hk, 'unsafe'), ('over_refusal', Tk, 'ref')]:
    for dim, a, b, keys in contrasts:
        for cat, g in list(d.groupby('category')) + [('ALL', d)]:
            n, up, dn, p, ra, rb = paired(g, dim, a, b, keys, val)
            rows.append(dict(metric=metric, contrast=f'{a} vs {b}', category=cat, pairs=n, a_rate=round(ra, 1), b_rate=round(rb, 1),
                             a_worse=up, b_worse=dn, p=round(p, 4), seeds=g.seed_id.nunique()))
P = pd.DataFrame(rows)
print('\n### Paired within-seed contrasts (a_worse = pair where a fails and b does not)')
print(P.to_string(index=False))
P.to_csv(f'{ROOT}/post_train_2/notebooks/combined/cross_category_paired.csv', index=False)

# ---- pool sizes per category available for a new experiment (analysis + heldout, harmful)
print('\n### Prompt pool sizes per category (all splits): harmful / twin, seeds')
print(df.groupby(['category', 'split']).agg(prompts=('prompt_id', 'count'), seeds=('seed_id', 'nunique')).unstack().to_string())
