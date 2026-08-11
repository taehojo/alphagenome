"""
Round-2 batch: tone audit excluding bibliography, R1-2 with Independence WC,
R1-5 with cluster-robust SE on gene, R1-4 with signed correlation.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Independence, Exchangeable

OUT = Path('/N/project/AiLab/alphagenome/code/phase10_revision_2026/CORRECTIONS_ROUND2/results')

from docx import Document
MS = Path('/N/project/AiLab/alphagenome/manuscript/circulate/Jo et al Rare_Variants_AD-Cell-05202026-v7g.docx')
doc = Document(MS)
paras = [p.text for p in doc.paragraphs if p.text.strip()]

bib_start = None
for i, p in enumerate(paras):
    if re.match(r'^references?\s*$', p, re.IGNORECASE):
        bib_start = i
        break
if bib_start is None:
    
    for i, p in enumerate(paras):
        if re.match(r'^\d+\.\s+[A-Z][a-z]', p) and i > 10:
            bib_start = i
            break
prose = paras[:bib_start] if bib_start else paras
print(f"Total paragraphs: {len(paras)}, prose paragraphs (excluding bibliography): {len(prose)}, bib_start: {bib_start}")
prose_text = '\n'.join(prose)

PROMOTIONAL = {
    'clinically_meaningful': r'\bclinically\s+meaningful\b',
    'strong_evidence': r'\bstrong\s+evidence\b',
    'robust_signal': r'\brobust\s+signal\b',
    'substantial_enrichment': r'\bsubstantial\s+enrichment\b',
    'striking': r'\bstriking\b',
    'remarkable': r'\bremarkable\b',
    'profound': r'\bprofound\b',
    'compelling': r'\bcompelling\b',
    'powerful': r'\bpowerful\b',
    'highly_significant': r'\bhighly\s+significant\b',
    'unprecedented': r'\bunprecedented\b',
    'novel': r'\bnovel\b',
    'first_to_show': r'\bfirst\s+to\s+(show|demonstrate|report)\b',
}
rows = []
for name, pat in PROMOTIONAL.items():
    m = re.findall(pat, prose_text, re.IGNORECASE)
    rows.append({'pattern': name, 'count': len(m)})
print("\nTone audit (prose only, promotional patterns only):")
tone = pd.DataFrame(rows).sort_values('count', ascending=False)
print(tone.to_string(index=False))
tone.to_csv(OUT / 'tone_audit_round2.csv', index=False)

df = pd.read_csv('/N/project/AiLab/alphagenome/code/phase10_revision_2026/R1-11_coding_noncoding/results/unique_9943_variants.csv')
df = df[np.isfinite(df['cc_ratio'])].copy()
df['case_enriched'] = (df['cc_ratio'] > 1).astype(int)
df = df.dropna(subset=['gene_name']).copy()

MODS = {'rna_seq_effect': 'RNA-seq', 'cage_effect': 'CAGE',
        'chip_histone_effect': 'ChIP-Histone', 'dnase_effect': 'DNase'}

def gee_two_struct(mod, cutoff):
    v = df[df[mod].notna()].copy()
    thresh = v[mod].quantile(cutoff)
    v['high'] = (v[mod] > thresh).astype(int)
    v = v.sort_values('gene_name').reset_index(drop=True)
    results = {}
    for struct, name in [(Exchangeable(), 'Exch'), (Independence(), 'Ind')]:
        try:
            m = smf.gee('case_enriched ~ high', groups='gene_name', data=v,
                        family=sm.families.Binomial(), cov_struct=struct)
            r = m.fit()
            results[name] = {
                'OR': np.exp(r.params['high']),
                'CI_lo': np.exp(r.params['high'] - 1.96*r.bse['high']),
                'CI_hi': np.exp(r.params['high'] + 1.96*r.bse['high']),
                'P': r.pvalues['high'],
            }
        except Exception as e:
            results[name] = {'OR': np.nan, 'CI_lo': np.nan, 'CI_hi': np.nan, 'P': np.nan}
    return results, len(v)

rows = []
for cut in [0.50, 0.80]:
    for mod, mname in MODS.items():
        r, n = gee_two_struct(mod, cut)
        rows.append({
            'modality': mname, 'cutoff': 'median' if cut==0.50 else 'top-20%', 'n': n,
            'Exch_OR': r['Exch']['OR'], 'Exch_P': r['Exch']['P'],
            'Exch_CI_lo': r['Exch']['CI_lo'], 'Exch_CI_hi': r['Exch']['CI_hi'],
            'Ind_OR': r['Ind']['OR'], 'Ind_P': r['Ind']['P'],
            'Ind_CI_lo': r['Ind']['CI_lo'], 'Ind_CI_hi': r['Ind']['CI_hi'],
        })
gee = pd.DataFrame(rows)
gee.to_csv(OUT / 'R1-2_GEE_both_WC.csv', index=False)
print("\nR1-2 GEE with both working correlations:")
print(gee.to_string(index=False))

def rint(x):
    n = len(x)
    ranks = pd.Series(x).rank(method='average').values
    return stats.norm.ppf((ranks - 0.5) / n)

print("\nR1-5 RINT regression with cluster-robust SE on gene:")
rint_rows = []
for mod, mname in MODS.items():
    v = df[df[mod].notna()].copy()
    v['z'] = rint(v[mod].values)
    v = v.sort_values('gene_name').reset_index(drop=True)
    
    X = sm.add_constant(v['z'])
    m_naive = sm.Logit(v['case_enriched'], X).fit(disp=False)
    
    m_cluster = sm.GLM(v['case_enriched'], X, family=sm.families.Binomial()).fit(
        cov_type='cluster', cov_kwds={'groups': v['gene_name'].values})
    rint_rows.append({
        'modality': mname,
        'OR_per_SD_naive': float(np.exp(m_naive.params['z'])),
        'P_naive': float(m_naive.pvalues['z']),
        'OR_per_SD_cluster': float(np.exp(m_cluster.params[1])),
        'P_cluster_gene': float(m_cluster.pvalues[1]),
        'CI_lo_cluster': float(np.exp(m_cluster.params[1] - 1.96*m_cluster.bse[1])),
        'CI_hi_cluster': float(np.exp(m_cluster.params[1] + 1.96*m_cluster.bse[1])),
    })
rint_df = pd.DataFrame(rint_rows)
rint_df.to_csv(OUT / 'R1-5_RINT_cluster_robust.csv', index=False)
print(rint_df.to_string(index=False))

g = pd.read_csv('/N/project/AiLab/alphagenome/code/phase10_revision_2026/R1-4_burden_benchmark/results/gene_level_burden_vs_IR.csv')
g['signed_log_burden_P'] = np.sign(g['burden_OR'] - 1) * (-np.log10(g['burden_P'].clip(lower=1e-300)))
sp_signed, p_signed = stats.spearmanr(g['IR_RNAseq'], g['signed_log_burden_P'])
sp_unsigned, p_unsigned = stats.spearmanr(g['IR_RNAseq'], -np.log10(g['burden_P'].clip(lower=1e-300)))
print(f"\nR1-4 SIGNED Spearman correlation IR vs sign(OR-1)*-log10(P): rho={sp_signed:.3f}, P={p_signed:.3f}")
print(f"R1-4 UNSIGNED Spearman (as before):  rho={sp_unsigned:.3f}, P={p_unsigned:.3f}")
pd.DataFrame([{
    'metric': 'unsigned_Spearman_rho', 'value': sp_unsigned, 'P': p_unsigned, 'n': len(g)
}, {
    'metric': 'signed_Spearman_rho', 'value': sp_signed, 'P': p_signed, 'n': len(g)
}]).to_csv(OUT / 'R1-4_signed_correlation.csv', index=False)
