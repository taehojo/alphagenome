import argparse
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

MODALITIES = [
    ("RNA-seq", "rna_seq_effect"),
    ("CAGE", "cage_effect"),
    ("ChIP-histone", "chip_histone_effect"),
    ("DNase", "dnase_effect"),
    ("Splice junctions", "splice_junctions_effect"),
    ("Splice sites", "splice_sites_effect"),
    ("Splice site usage", "splice_site_usage_effect"),
    ("ChIP-TF", "chip_tf_effect"),
]
PRIMARY = [m for m in MODALITIES[:4]]
BONFERRONI = 0.05 / 8


def analysis_set(df, min_ac=3):
    df = df.assign(total_AC=df.case_AC + df.ctrl_AC)
    df = df[df.total_AC >= min_ac]
    df = df.sort_values("total_AC", ascending=False).drop_duplicates("variant_id", keep="first")
    df = df.dropna(subset=[c for _, c in PRIMARY])
    return df[np.isfinite(df.cc_ratio)]


def interaction_ratio(sub, column, pct=80):
    score = sub[column].values
    case_enriched = (sub.cc_ratio > 1).values
    high = score >= np.percentile(score, pct)
    a, b = case_enriched[high].sum(), (~case_enriched[high]).sum()
    c, d = case_enriched[~high].sum(), (~case_enriched[~high]).sum()
    hi_pct, lo_pct = a / (a + b) * 100, c / (c + d) * 100
    odds, p = fisher_exact([[a, b], [c, d]])
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(n=len(sub), n_high=int(high.sum()), high_pct=hi_pct, low_pct=lo_pct,
                ir=hi_pct / lo_pct, p=p, odds_ratio=odds,
                or_lo=np.exp(np.log(odds) - 1.96 * se), or_hi=np.exp(np.log(odds) + 1.96 * se))


def median_split(sub, column):
    score = sub[column].values
    case_enriched = (sub.cc_ratio > 1).values
    high = score > np.median(score)
    a, b = case_enriched[high].sum(), (~case_enriched[high]).sum()
    c, d = case_enriched[~high].sum(), (~case_enriched[~high]).sum()
    hi_pct, lo_pct = a / (a + b) * 100, c / (c + d) * 100
    return hi_pct / lo_pct, fisher_exact([[a, b], [c, d]])[1]


def table2_s5(df):
    rows = []
    for name, column in MODALITIES:
        if column not in df.columns:
            continue
        r = interaction_ratio(df, column)
        rows.append(dict(modality=name, n=r["n"], high_effect_pct=round(r["high_pct"], 1),
                         low_effect_pct=round(r["low_pct"], 1), ir=round(r["ir"], 3),
                         fisher_p=r["p"], odds_ratio=round(r["odds_ratio"], 2),
                         or_ci_low=round(r["or_lo"], 2), or_ci_high=round(r["or_hi"], 2),
                         passes_bonferroni=r["p"] < BONFERRONI))
    return pd.DataFrame(rows)


def s7_panel_a(raw):
    rows = []
    for ac in (1, 3, 5, 10):
        sub = analysis_set(raw, ac)
        pct_case = (sub.cc_ratio > 1).mean() * 100
        for name, column in PRIMARY:
            r = interaction_ratio(sub, column)
            rows.append(dict(allele_count=f">={ac}", modality=name, n=r["n"],
                             case_enriched_pct=round(pct_case, 1),
                             ir=round(r["ir"], 4), fisher_p=r["p"]))
    return pd.DataFrame(rows)


def s7_panel_b(df):
    rows = []
    for pct in range(50, 100, 5):
        r = interaction_ratio(df, "rna_seq_effect", pct)
        rows.append(dict(percentile=pct, top_pct=f"{100 - pct}%", n_high=r["n_high"],
                         n_low=r["n"] - r["n_high"], ir=round(r["ir"], 4), fisher_p=r["p"]))
    return pd.DataFrame(rows)


def s7_panel_c(df):
    score = df.rna_seq_effect.values
    case_enriched = (df.cc_ratio > 1).values
    decile = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False).values
    rows, exact = [], []
    for i in range(10):
        m = decile == i
        pct = case_enriched[m].mean() * 100
        exact.append(pct)
        rows.append(dict(decile=i + 1, n=int(m.sum()), case_enriched_pct=round(pct, 1),
                         mean_effect=round(score[m].mean(), 4)))
    out = pd.DataFrame(rows)
    rho, p = spearmanr(range(1, 11), exact)
    variant_rho = spearmanr(score, case_enriched.astype(int))[0]
    out.attrs["spearman_r"], out.attrs["spearman_p"] = round(rho, 3), round(p, 3)
    out.attrs["variant_level_r"] = round(variant_rho, 3)
    return out


def s8(df):
    rows = []
    for name, column in PRIMARY:
        top = interaction_ratio(df, column)
        med_ir, med_p = median_split(df, column)
        rows.append(dict(modality=name, dichotomisation="Top 20%", n=top["n"],
                         ir=round(top["ir"], 3), fisher_p=top["p"],
                         passes_bonferroni=top["p"] < BONFERRONI))
        rows.append(dict(modality=name, dichotomisation="Median split", n=len(df),
                         ir=round(med_ir, 3), fisher_p=med_p,
                         passes_bonferroni=med_p < BONFERRONI))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    raw = pd.read_csv(args.variants)
    df = analysis_set(raw)
    print(f"analysis set: n = {len(df)}")

    t2 = table2_s5(df)
    pa, pb, pc, t8 = s7_panel_a(raw), s7_panel_b(df), s7_panel_c(df), s8(df)

    t2.to_csv(f"{args.outdir}/table2_S5_interaction_ratio.csv", index=False)
    pa.to_csv(f"{args.outdir}/S7A_allele_count_threshold.csv", index=False)
    pb.to_csv(f"{args.outdir}/S7B_percentile_cutoff.csv", index=False)
    pc.to_csv(f"{args.outdir}/S7C_decile_dose_response.csv", index=False)
    t8.to_csv(f"{args.outdir}/S8_cutoff_comparison.csv", index=False)

    print(t2.to_string(index=False))
    print(f"\ndecile Spearman r = {pc.attrs['spearman_r']}, P = {pc.attrs['spearman_p']}; "
          f"variant-level r = {pc.attrs['variant_level_r']}")


if __name__ == "__main__":
    main()
