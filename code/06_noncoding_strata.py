"""
R1-11: Re-compute Interaction Ratio (IR) per modality in 3 strata:
  (a) ALL                 = all 9,974 unique variants
  (b) NO PROTEIN-ALTERING = exclude missense, nonsense, splice donor/acceptor, frameshift, etc.
  (c) NON-CODING ONLY     = only non-coding (drop both protein-altering AND synonymous)

IR definition (matches phase5 figure2_modality_effects.py panel C):
  threshold = median of valid modality scores (over current stratum)
  high group: score > median
  IR = (% case-enriched in high) / (% case-enriched in low)
  95% CI by 1000 bootstrap resamples
  P-value by permutation test of case_enriched labels (10000 iter)

case_enriched = (cc_ratio > 1) = (case_AF > ctrl_AF)
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/N/project/AiLab/alphagenome/code/phase10_revision_2026/R1-11_coding_noncoding')

variants = pd.read_csv(BASE / 'results' / 'unique_9943_variants.csv')
annot = pd.read_csv(BASE / 'results' / 'variants_annotated.csv')

df = variants.merge(annot[['variant_id', 'broad_class', 'display_class']],
                    on='variant_id', how='left')
print(f"Merged: {len(df):,} variants (missing annot: {df['broad_class'].isna().sum()})")

df = df[np.isfinite(df['cc_ratio'])].copy()
df['case_enriched'] = (df['cc_ratio'] > 1).astype(int)
print(f"Finite cc_ratio: {len(df):,}, case-enriched: {df['case_enriched'].sum():,} "
      f"({100*df['case_enriched'].mean():.1f}%)")

MODALITIES = {
    'rna_seq_effect':       'RNA-seq',
    'cage_effect':          'CAGE',
    'chip_histone_effect':  'ChIP-Histone',
    'dnase_effect':         'DNase',
}

def compute_ir(sub, mod, n_boot=1000, n_perm=10000, seed=42):
    """Median-split IR + bootstrap 95% CI + permutation p-value."""
    v = sub[sub[mod].notna()].copy()
    if len(v) < 50:
        return None
    med = v[mod].median()
    high = v[mod] > med
    h_pct = v.loc[high, 'case_enriched'].mean() * 100
    l_pct = v.loc[~high, 'case_enriched'].mean() * 100
    ir = h_pct / l_pct if l_pct > 0 else np.nan

    rng = np.random.default_rng(seed)
    irs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(v), size=len(v))
        s = v.iloc[idx]
        h = s[mod] > med
        hp = s.loc[h, 'case_enriched'].mean() * 100
        lp = s.loc[~h, 'case_enriched'].mean() * 100
        if lp > 0:
            irs.append(hp / lp)
    ci_lo, ci_hi = np.percentile(irs, [2.5, 97.5])

    
    null_irs = []
    case_lab = v['case_enriched'].values.copy()
    score_high = (v[mod] > med).values
    for _ in range(n_perm):
        rng.shuffle(case_lab)
        hp = case_lab[score_high].mean() * 100
        lp = case_lab[~score_high].mean() * 100
        if lp > 0:
            null_irs.append(hp / lp)
    null_irs = np.array(null_irs)
    
    p_emp = (np.abs(null_irs - 1) >= np.abs(ir - 1)).mean()

    return {
        'n_total': len(v),
        'n_case_enriched': int(v['case_enriched'].sum()),
        'pct_case_enriched': float(100 * v['case_enriched'].mean()),
        'median_effect': float(med),
        'IR': float(ir),
        'CI_low': float(ci_lo),
        'CI_high': float(ci_hi),
        'high_pct': float(h_pct),
        'low_pct': float(l_pct),
        'p_permutation': float(p_emp),
    }

strata = {
    'ALL':                df.copy(),
    'NO_PROTEIN_ALTERING': df[df['broad_class'] != 'PROTEIN_ALTERING'].copy(),
    'NON_CODING_ONLY':    df[df['broad_class'] == 'NON_CODING'].copy(),
}

rows = []
for sname, sub in strata.items():
    print(f"\n=== Stratum: {sname}  (n={len(sub):,}) ===")
    for mod, mname in MODALITIES.items():
        r = compute_ir(sub, mod)
        if r is None:
            print(f"  {mname}: insufficient")
            continue
        r['stratum'] = sname
        r['modality'] = mname
        r['n_stratum'] = len(sub)
        rows.append(r)
        print(f"  {mname:12s}: IR = {r['IR']:.3f} "
              f"[{r['CI_low']:.3f}-{r['CI_high']:.3f}], P = {r['p_permutation']:.2e}, "
              f"n_modality_valid = {r['n_total']:,}")

out = pd.DataFrame(rows)
out_path = BASE / 'results' / 'IR_3strata.csv'
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

strat_summary = pd.DataFrame([
    {'stratum': k, 'n_variants': len(v), 'pct_case_enriched': 100*v['case_enriched'].mean()}
    for k, v in strata.items()
])
strat_summary.to_csv(BASE / 'results' / 'stratum_sizes.csv', index=False)
print(strat_summary.to_string(index=False))
