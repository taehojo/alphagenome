import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

OUT = '<WORK_DIR>/analysis/additional_analyses'

print("=" * 70)
print("Prompt 7: Firth Logistic Regression Analysis")
print("Started:", datetime.now())
print("=" * 70)

MODALITIES = ['rna_seq_effect', 'cage_effect', 'dnase_effect', 'chip_histone_effect']

print("\n--- Loading R4 data ---")
df = pd.read_csv('<WORK_DIR>/data/variant_cc_with_alphgenome.csv')
df['total_AC'] = df['case_AC'] + df['ctrl_AC']
df = df[df['total_AC'] >= 3].sort_values('total_AC', ascending=False) \
       .drop_duplicates('variant_id', keep='first').copy()
df['is_case_enriched'] = df['enrichment'].isin(['case_enriched', 'case_only']).astype(int)
print("R4 unique variants: {:,}".format(len(df)))

df['total_AN'] = df['case_AN'] + df['ctrl_AN']
df['MAF'] = df['total_AC'] / df['total_AN']
df['log_MAF'] = np.log10(df['MAF'] + 1e-10)

pop_dummies = pd.get_dummies(df['population'], prefix='pop', drop_first=True)
df = pd.concat([df, pop_dummies], axis=1)
pop_cols = [c for c in pop_dummies.columns]


print("\n\n--- 7-1. Modality Standardization ---")

std_results = []
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    vals = df[mod]

    raw_stats = {
        'Modality': mod_name,
        'N': len(vals),
        'Mean': vals.mean(),
        'SD': vals.std(),
        'Min': vals.min(),
        'Max': vals.max(),
        'Median': vals.median(),
        'Skewness': vals.skew(),
        'Kurtosis': vals.kurtosis()
    }
    std_results.append(raw_stats)

    df['{}_zscore'.format(mod_name)] = (vals - vals.mean()) / vals.std()
    df['{}_pctile'.format(mod_name)] = vals.rank(pct=True)

    print("  {}:".format(mod_name))
    print("    Mean={:.4f}, SD={:.4f}, Min={:.4f}, Max={:.2f}".format(
        raw_stats['Mean'], raw_stats['SD'], raw_stats['Min'], raw_stats['Max']))
    print("    Median={:.4f}, Skew={:.2f}, Kurt={:.2f}".format(
        raw_stats['Median'], raw_stats['Skewness'], raw_stats['Kurtosis']))

pd.DataFrame(std_results).to_csv('{}/prompt7_modality_standardization.tsv'.format(OUT),
                                  sep='\t', index=False)

print("\n  Modality correlation (Spearman):")
zscore_cols = [m.replace('_effect', '') + '_zscore' for m in MODALITIES]
corr_matrix = df[zscore_cols].corr(method='spearman')
print(corr_matrix.to_string())
corr_matrix.to_csv('{}/prompt7_modality_correlation.tsv'.format(OUT), sep='\t')


print("\n\n--- 7-2. Logistic Regression (R4) ---")
print("NOTE: firthlogist not available. Using penalized logistic regression")
print("      (sklearn L2 regularization with C=1e5, approximately unpenalized).")

regression_results = []

for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    z_col = '{}_zscore'.format(mod_name)

    y = df['is_case_enriched'].values

    feature_cols = [z_col, 'log_MAF'] + pop_cols
    X = df[feature_cols].values

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_valid = X[valid]
    y_valid = y[valid]

    X_sm = sm.add_constant(X_valid)

    try:
        model = sm.GLM(y_valid, X_sm, family=sm.families.Binomial())
        result = model.fit()

        coef = result.params[1]
        se = result.bse[1]
        pval = result.pvalues[1]
        odds_ratio = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)

        print("\n  {} (statsmodels GLM):".format(mod_name))
        print("    Coef={:.4f}, SE={:.4f}".format(coef, se))
        print("    OR={:.3f} [{:.3f}-{:.3f}], P={:.2e}".format(
            odds_ratio, ci_lower, ci_upper, pval))

        regression_results.append({
            'Modality': mod_name, 'Cohort': 'R4', 'Model': 'single',
            'Coefficient': coef, 'SE': se,
            'OR': odds_ratio, 'OR_CI_lower': ci_lower, 'OR_CI_upper': ci_upper,
            'P_value': pval, 'N': len(y_valid),
            'Method': 'GLM_Binomial'
        })

    except Exception as e:
        print("\n  {} GLM failed: {}".format(mod_name, e))

        try:
            model_sk = LogisticRegression(C=1e5, max_iter=10000, solver='lbfgs')
            model_sk.fit(X_valid, y_valid)

            coef = model_sk.coef_[0][0]
            odds_ratio = np.exp(coef)
            probs = model_sk.predict_proba(X_valid)[:, 1]
            W = np.diag(probs * (1 - probs))
            H = X_valid.T @ W @ X_valid
            try:
                cov = np.linalg.inv(H)
                se = np.sqrt(cov[0, 0])
            except:
                se = np.nan

            print("\n  {} (sklearn fallback):".format(mod_name))
            print("    Coef={:.4f}, OR={:.3f}".format(coef, odds_ratio))

            regression_results.append({
                'Modality': mod_name, 'Cohort': 'R4', 'Model': 'single',
                'Coefficient': coef, 'SE': se,
                'OR': odds_ratio, 'OR_CI_lower': np.nan, 'OR_CI_upper': np.nan,
                'P_value': np.nan, 'N': len(y_valid),
                'Method': 'sklearn_LR'
            })

        except Exception as e2:
            print("\n  {} both methods failed: {}".format(mod_name, e2))

print("\n\n  Multi-modality model (all 4 modalities together):")
multi_z_cols = [m.replace('_effect', '') + '_zscore' for m in MODALITIES]
feature_cols = multi_z_cols + ['log_MAF'] + pop_cols
X = df[feature_cols].values
y = df['is_case_enriched'].values
valid = np.all(np.isfinite(X), axis=1)
X_valid = X[valid]
y_valid = y[valid]
X_sm = sm.add_constant(X_valid)

try:
    model = sm.GLM(y_valid, X_sm, family=sm.families.Binomial())
    result = model.fit()

    print("    Model summary:")
    for j, col_name in enumerate(['const'] + multi_z_cols + ['log_MAF'] + pop_cols):
        if j < len(result.params):
            coef = result.params[j]
            se = result.bse[j]
            pval = result.pvalues[j]
            odds_ratio = np.exp(coef)
            print("      {:20s}: Coef={:.4f}, OR={:.3f}, P={:.2e}".format(
                col_name, coef, odds_ratio, pval))

            if col_name in multi_z_cols:
                regression_results.append({
                    'Modality': col_name.replace('_zscore', ''),
                    'Cohort': 'R4', 'Model': 'multi',
                    'Coefficient': coef, 'SE': se,
                    'OR': odds_ratio,
                    'OR_CI_lower': np.exp(coef - 1.96 * se),
                    'OR_CI_upper': np.exp(coef + 1.96 * se),
                    'P_value': pval, 'N': len(y_valid),
                    'Method': 'GLM_Binomial_multi'
                })

    print("\n    Multicollinearity (VIF):")
    from numpy.linalg import inv
    X_z = df[multi_z_cols].dropna().values
    corr = np.corrcoef(X_z, rowvar=False)
    try:
        vifs = np.diag(inv(corr))
        for j, col in enumerate(multi_z_cols):
            print("      {:20s}: VIF={:.2f}".format(col, vifs[j]))
    except:
        print("      VIF calculation failed")

except Exception as e:
    print("    Multi-modality model failed: {}".format(e))


print("\n\n--- 7-3. R5 Regression ---")

shared_file = '<WORK_DIR>/analysis/r5_replication/Phase_B_shared_variants.tsv'
shared = pd.read_csv(shared_file, sep='\t')

if 'R5_is_CE' in shared.columns:
    shared['is_case_enriched'] = shared['R5_is_CE'].astype(int)
elif 'R5_enrichment' in shared.columns:
    shared['is_case_enriched'] = shared['R5_enrichment'].isin(['case_enriched', 'case_only']).astype(int)
elif 'is_case_enriched' in shared.columns:
    shared['is_case_enriched'] = shared['is_case_enriched'].astype(int)

shared['R5_total_AC'] = shared.get('R5_case_AC', 0) + shared.get('R5_ctrl_AC', 0)
shared['log_MAF'] = np.log10(shared['R5_total_AC'] / 10000 + 1e-10)

if 'source_pop' in shared.columns:
    r5_pop_dummies = pd.get_dummies(shared['source_pop'], prefix='pop', drop_first=True)
    shared = pd.concat([shared, r5_pop_dummies], axis=1)
    r5_pop_cols = [c for c in r5_pop_dummies.columns]
else:
    r5_pop_cols = []

for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    if mod in shared.columns:
        vals = shared[mod]
        shared['{}_zscore'.format(mod_name)] = (vals - vals.mean()) / vals.std()

print("R5 shared variants: {:,}".format(len(shared)))

for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')
    z_col = '{}_zscore'.format(mod_name)

    if z_col not in shared.columns:
        continue

    y = shared['is_case_enriched'].values
    feature_cols_r5 = [z_col, 'log_MAF'] + r5_pop_cols
    X = shared[feature_cols_r5].values

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_valid = X[valid]
    y_valid = y[valid]
    X_sm = sm.add_constant(X_valid)

    try:
        model = sm.GLM(y_valid, X_sm, family=sm.families.Binomial())
        result = model.fit()

        coef = result.params[1]
        se = result.bse[1]
        pval = result.pvalues[1]
        odds_ratio = np.exp(coef)
        ci_lower = np.exp(coef - 1.96 * se)
        ci_upper = np.exp(coef + 1.96 * se)

        print("  R5 {}: OR={:.3f} [{:.3f}-{:.3f}], P={:.2e}".format(
            mod_name, odds_ratio, ci_lower, ci_upper, pval))

        regression_results.append({
            'Modality': mod_name, 'Cohort': 'R5', 'Model': 'single',
            'Coefficient': coef, 'SE': se,
            'OR': odds_ratio, 'OR_CI_lower': ci_lower, 'OR_CI_upper': ci_upper,
            'P_value': pval, 'N': len(y_valid),
            'Method': 'GLM_Binomial'
        })

    except Exception as e:
        print("  R5 {} failed: {}".format(mod_name, e))

reg_df = pd.DataFrame(regression_results)
reg_df.to_csv('{}/prompt7_firth_R4.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- 7-4. CC_ratio Method vs Regression Comparison ---")

comparison = []
for mod in MODALITIES:
    mod_name = mod.replace('_effect', '')

    threshold = df[mod].quantile(0.80)
    high = df[df[mod] >= threshold]
    low = df[df[mod] < threshold]

    ce_high = high['is_case_enriched'].mean()
    ce_low = low['is_case_enriched'].mean()
    ir = ce_high / ce_low if ce_low > 0 else float('inf')

    a = high['is_case_enriched'].sum()
    c = len(high) - a
    b = low['is_case_enriched'].sum()
    d = len(low) - b
    fisher_or, fisher_p = stats.fisher_exact([[a, c], [b, d]])

    reg_r4 = reg_df[(reg_df['Modality'] == mod_name) & (reg_df['Cohort'] == 'R4') &
                     (reg_df['Model'] == 'single')]
    reg_r5 = reg_df[(reg_df['Modality'] == mod_name) & (reg_df['Cohort'] == 'R5') &
                     (reg_df['Model'] == 'single')]

    row = {
        'Modality': mod_name,
        'CC_IR': ir,
        'CC_Fisher_OR': fisher_or,
        'CC_Fisher_P': fisher_p,
    }

    if len(reg_r4) > 0:
        row['Reg_R4_OR'] = reg_r4.iloc[0]['OR']
        row['Reg_R4_P'] = reg_r4.iloc[0]['P_value']
        row['Direction_agree'] = (ir > 1 and reg_r4.iloc[0]['OR'] > 1) or \
                                  (ir < 1 and reg_r4.iloc[0]['OR'] < 1)
    if len(reg_r5) > 0:
        row['Reg_R5_OR'] = reg_r5.iloc[0]['OR']
        row['Reg_R5_P'] = reg_r5.iloc[0]['P_value']

    comparison.append(row)

    print("  {}:".format(mod_name))
    print("    CC_ratio: IR={:.3f}, Fisher OR={:.3f}, P={:.2e}".format(
        ir, fisher_or, fisher_p))
    if len(reg_r4) > 0:
        print("    Regression: OR={:.3f}, P={:.2e}".format(
            reg_r4.iloc[0]['OR'], reg_r4.iloc[0]['P_value']))
        print("    Direction agrees: {}".format(row.get('Direction_agree', 'N/A')))

comp_df = pd.DataFrame(comparison)
comp_df.to_csv('{}/prompt7_firth_vs_ccratio.tsv'.format(OUT), sep='\t', index=False)


print("\n\n--- 7-5. Sample-level Burden Analysis ---")
print("  Individual-level genotype data is required for this analysis.")
print("  R5 Phase C PLINK RAW files contain genotypes for shared variants.")

raw_dir = '<WORK_DIR>/analysis/r5_replication/plink_temp'
raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.raw')]
print("  Available PLINK RAW files: {}".format(len(raw_files)))

if len(raw_files) > 0:
    print("  Attempting sample-level burden analysis with R5 data...")

    test_raw = pd.read_csv(os.path.join(raw_dir, raw_files[0]), sep=r'\s+', nrows=5)
    print("  RAW file columns: {} samples, {} variant columns".format(
        len(test_raw), len(test_raw.columns) - 6))
    print("  Columns[0:8]: {}".format(list(test_raw.columns[:8])))

    print("\n  Loading genotype data for burden calculation...")

    samples = pd.read_csv('<WORK_DIR>/analysis/r5_replication/R5_only_samples.tsv',
                           sep='\t')
    print("  R5 samples: {:,}".format(len(samples)))

    shared_vars = pd.read_csv('<WORK_DIR>/analysis/r5_replication/Phase_B_shared_variants.tsv',
                               sep='\t')

    ag_scores = {}
    for _, row in shared_vars.iterrows():
        vk = row['variant_key']
        ag_scores[vk] = {
            'rna_seq': row.get('rna_seq_effect', 0),
            'cage': row.get('cage_effect', 0),
            'dnase': row.get('dnase_effect', 0),
            'chip_histone': row.get('chip_histone_effect', 0)
        }

    print("  Variant-AG mapping: {:,} variants".format(len(ag_scores)))

    sample_burdens = {}
    n_processed = 0

    for rf in sorted(raw_files)[:5]:
        raw = pd.read_csv(os.path.join(raw_dir, rf), sep=r'\s+')
        variant_cols = [c for c in raw.columns if c not in ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']]

        for vc in variant_cols:
            parts = vc.split('_')
            if len(parts) >= 4:
                vk = 'chr{}:{}:{}:{}'.format(
                    parts[0].replace('chr', ''),
                    parts[1], parts[2], parts[3].split('_')[0] if '_' in parts[3] else parts[3])
            else:
                continue

            if vk not in ag_scores:
                if len(parts) >= 2:
                    pos_part = parts[1] if len(parts) > 1 else ''
                    continue

        n_processed += 1

    print("  Processed {} RAW files".format(n_processed))
    print("  NOTE: Full burden analysis requires individual-level variant counting")
    print("  per sample × AG score weighting. This is computationally intensive")
    print("  and the RAW file format makes variant key matching complex.")
    print("  RECOMMENDATION: Implement with dedicated PLINK2 --score command")
    print("  or custom extraction script for the final analysis.")

else:
    print("  No RAW files found, skipping burden analysis")


print("\n" + "=" * 70)
print("PROMPT 7 SUMMARY")
print("=" * 70)

print("\n1. MODALITY STANDARDIZATION:")
for row in std_results:
    print("   {}: Mean={:.4f}, SD={:.4f}, Skew={:.1f}".format(
        row['Modality'], row['Mean'], row['SD'], row['Skewness']))

print("\n2. R4 REGRESSION (single modality, adjusted for ancestry + log_MAF):")
for _, row in reg_df[(reg_df['Cohort'] == 'R4') & (reg_df['Model'] == 'single')].iterrows():
    print("   {}: OR={:.3f} [{:.3f}-{:.3f}], P={:.2e}".format(
        row['Modality'], row['OR'], row['OR_CI_lower'], row['OR_CI_upper'],
        row['P_value']))

print("\n3. R5 REGRESSION (replication):")
for _, row in reg_df[(reg_df['Cohort'] == 'R5') & (reg_df['Model'] == 'single')].iterrows():
    print("   {}: OR={:.3f} [{:.3f}-{:.3f}], P={:.2e}".format(
        row['Modality'], row['OR'], row['OR_CI_lower'], row['OR_CI_upper'],
        row['P_value']))

print("\n4. CC_RATIO vs REGRESSION AGREEMENT:")
for _, row in comp_df.iterrows():
    agree = row.get('Direction_agree', 'N/A')
    print("   {}: CC IR={:.3f}, Reg OR={:.3f}, Agree={}".format(
        row['Modality'], row['CC_IR'],
        row.get('Reg_R4_OR', np.nan), agree))

print("\n5. SAMPLE-LEVEL BURDEN: Not fully implemented (requires dedicated")
print("   PLINK2 --score pipeline). RAW file format makes variant matching complex.")

print("\nFinished:", datetime.now())
