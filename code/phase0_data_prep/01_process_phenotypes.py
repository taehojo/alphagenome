import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

WORK_DIR = "<WORK_DIR>"
ADSP_DATA_BASE = "<PLINK_DATA_DIR>"
ADSP_PHENOTYPE = "<PHENOTYPE_FILE>"
POPULATION_PREFIXES = ["AA", "NHW", "Asian", "Hispanic"]

def process_phenotypes_and_ids():
    print(f"[{datetime.now()}] Starting phenotype processing")
    print(f"[{datetime.now()}] Phenotype file: {ADSP_PHENOTYPE}")

    if not os.path.exists(ADSP_PHENOTYPE):
        print(f"ERROR: Phenotype file not found: {ADSP_PHENOTYPE}")
        sys.exit(1)

    print(f"[{datetime.now()}] Loading phenotype data...")
    pheno_df = pd.read_csv(ADSP_PHENOTYPE)

    print(f"[{datetime.now()}] Total samples in phenotype file: {len(pheno_df):,}")
    print(f"[{datetime.now()}] Columns: {list(pheno_df.columns)}")

    ad_status_col = None
    for col in ['AD_status', 'AD', 'Status', 'Phenotype']:
        if col in pheno_df.columns:
            ad_status_col = col
            break

    if not ad_status_col:
        print("WARNING: No AD status column found. Available columns:")
        print(pheno_df.columns.tolist())
        ad_status_col = 'AD_status'

    all_matched_samples = []

    for population in POPULATION_PREFIXES:
        fam_file = f"{ADSP_DATA_BASE}/{population}.fam"

        print(f"\n[{datetime.now()}] Processing {population} population...")
        print(f"[{datetime.now()}] FAM file: {fam_file}")

        if not os.path.exists(fam_file):
            print(f"WARNING: FAM file not found: {fam_file}")
            continue

        fam_df = pd.read_csv(fam_file, sep=' ', header=None,
                            names=['FID', 'IID', 'Father', 'Mother', 'Sex', 'Phenotype'])

        print(f"[{datetime.now()}] Samples in {population} FAM file: {len(fam_df):,}")


        direct_matches = fam_df[fam_df['IID'].isin(pheno_df['SUBJID'])]

        if len(direct_matches) > 0:
            print(f"[{datetime.now()}] Direct ID matches found: {len(direct_matches):,}")

        fam_df['short_id'] = fam_df['IID'].apply(lambda x: '-'.join(str(x).split('-')[:3]))

        matched = fam_df.merge(pheno_df, left_on='short_id', right_on='SUBJID', how='inner')

        print(f"[{datetime.now()}] Matched samples after ID processing: {len(matched):,}")

        if len(matched) == 0:
            print(f"[{datetime.now()}] Attempting alternative ID matching...")

            matched = fam_df.merge(pheno_df, left_on='FID', right_on='SUBJID', how='inner')

            if len(matched) == 0:
                print(f"WARNING: No samples matched for {population}")
                continue

        matched['Population'] = population

        if ad_status_col in matched.columns:
            cases = matched[matched[ad_status_col] == 1]
            controls = matched[matched[ad_status_col] == 0]
            print(f"[{datetime.now()}] Cases: {len(cases):,}, Controls: {len(controls):,}")

        all_matched_samples.append(matched)

    if all_matched_samples:
        combined_df = pd.concat(all_matched_samples, ignore_index=True)
        print(f"\n[{datetime.now()}] Total matched samples across all populations: {len(combined_df):,}")

        output_file = f"{WORK_DIR}/matched_samples.tsv"
        combined_df.to_csv(output_file, sep='\t', index=False)
        print(f"[{datetime.now()}] Matched samples saved to: {output_file}")

        plink_pheno_file = f"{WORK_DIR}/plink_phenotype.txt"

        plink_pheno = combined_df[['FID', 'IID']].copy()

        if ad_status_col in combined_df.columns:
            plink_pheno['AD'] = combined_df[ad_status_col].map({0: 1, 1: 2, -9: -9})
        else:
            plink_pheno['AD'] = combined_df['Phenotype']

        plink_pheno.to_csv(plink_pheno_file, sep='\t', header=False, index=False)
        print(f"[{datetime.now()}] PLINK phenotype file saved to: {plink_pheno_file}")

        print("\n=== Summary Statistics ===")
        print(f"Total unique samples: {len(combined_df):,}")

        for pop in POPULATION_PREFIXES:
            pop_data = combined_df[combined_df['Population'] == pop]
            if len(pop_data) > 0:
                print(f"\n{pop} population:")
                print(f"  Total: {len(pop_data):,}")

                if ad_status_col in pop_data.columns:
                    cases = len(pop_data[pop_data[ad_status_col] == 1])
                    controls = len(pop_data[pop_data[ad_status_col] == 0])
                    print(f"  Cases: {cases:,}")
                    print(f"  Controls: {controls:,}")
                    if controls > 0:
                        print(f"  Case/Control ratio: {cases/controls:.2f}")

        return combined_df
    else:
        print("ERROR: No matched samples found across any population")
        sys.exit(1)

def create_population_specific_phenotypes():
    print(f"\n[{datetime.now()}] Creating population-specific phenotype files...")

    matched_df = pd.read_csv(f"{WORK_DIR}/matched_samples.tsv", sep='\t')

    for population in POPULATION_PREFIXES:
        pop_samples = matched_df[matched_df['Population'] == population]

        if len(pop_samples) > 0:
            pop_pheno_file = f"{WORK_DIR}/{population}_phenotype.txt"

            plink_format = pop_samples[['FID', 'IID']].copy()

            ad_col = None
            for col in pop_samples.columns:
                if 'AD' in col.upper() or 'STATUS' in col.upper():
                    ad_col = col
                    break

            if ad_col and ad_col in pop_samples.columns:
                plink_format['AD'] = pop_samples[ad_col].map({0: 1, 1: 2}).fillna(-9)
            else:
                plink_format['AD'] = -9

            plink_format.to_csv(pop_pheno_file, sep='\t', header=False, index=False)
            print(f"[{datetime.now()}] {population} phenotype file: {pop_pheno_file} ({len(pop_samples):,} samples)")

if __name__ == "__main__":
    print("=" * 80)
    print("ADSP Genome-Wide Rare Variant Analysis")
    print("Phase 1A: Process Phenotypes and ID Matching")
    print("=" * 80)

    matched_samples = process_phenotypes_and_ids()

    create_population_specific_phenotypes()

    print(f"\n[{datetime.now()}] Phase 1A completed successfully")
    print("Next step: Run 02_extract_rare_variants.py")