import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
import multiprocessing
import time
import json

WORK_DIR = "<WORK_DIR>"
ADSP_DATA_BASE = "<PLINK_DATA_DIR>"
POPULATION_PREFIXES = ["AA", "NHW", "Asian", "Hispanic"]

THREADS = 8
MAF_THRESHOLD = 0.01

def save_population_checkpoint(population, status):
    checkpoint_file = f"{WORK_DIR}/.phase1b_checkpoint.json"

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {'populations': {}}

    checkpoint['populations'][population] = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def run_plink_rare_variant_extraction(population):
    print(f"[{datetime.now()}] Starting rare variant extraction for {population}")

    checkpoint_file = f"{WORK_DIR}/.phase1b_checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            if checkpoint.get('populations', {}).get(population, {}).get('status') == 'completed':
                result_file = f"{WORK_DIR}/results/{population}/{population}_rare_variant_list.tsv"
                if os.path.exists(result_file):
                    print(f"[{datetime.now()}] {population} already completed, skipping...")
                    return -1

    bed_file = f"{ADSP_DATA_BASE}/{population}.bed"
    bim_file = f"{ADSP_DATA_BASE}/{population}.bim"
    fam_file = f"{ADSP_DATA_BASE}/{population}.fam"
    pheno_file = f"{WORK_DIR}/{population}_phenotype.txt"

    for file_path in [bed_file, bim_file, fam_file]:
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            return None

    if not os.path.exists(pheno_file):
        print(f"WARNING: Phenotype file not found: {pheno_file}")
        print("Using general phenotype file instead")
        pheno_file = f"{WORK_DIR}/plink_phenotype.txt"

    output_dir = f"{WORK_DIR}/results/{population}"
    os.makedirs(output_dir, exist_ok=True)

    fam_df = pd.read_csv(fam_file, sep=' ', header=None)
    n_samples = len(fam_df)
    print(f"[{datetime.now()}] {population}: {n_samples:,} samples")

    mac_threshold = max(1, int(n_samples * 2 * MAF_THRESHOLD))
    print(f"[{datetime.now()}] MAC threshold for MAF < {MAF_THRESHOLD}: {mac_threshold}")

    print(f"[{datetime.now()}] Step 1: Extracting rare variants...")

    plink_rare_cmd = [
        'plink2',
        '--bed', bed_file,
        '--bim', bim_file,
        '--fam', fam_file,
        '--mac', '1',
        '--max-mac', str(mac_threshold),
        '--make-bed',
        '--out', f"{output_dir}/{population}_rare_variants",
        '--threads', str(THREADS)
    ]

    try:
        result = subprocess.run(plink_rare_cmd, capture_output=True, text=True, check=True)
        print(f"[{datetime.now()}] Rare variant extraction completed for {population}")

        for line in result.stdout.split('\n'):
            if 'variants' in line.lower():
                print(f"[{datetime.now()}] {line.strip()}")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: PLINK failed for {population}")
        print(f"STDERR: {e.stderr}")
        return None

    print(f"[{datetime.now()}] Step 2: Running association analysis...")

    plink_assoc_cmd = [
        'plink2',
        '--bfile', f"{output_dir}/{population}_rare_variants",
        '--pheno', pheno_file,
        '--glm', 'firth',
        '--ci', '0.95',
        '--out', f"{output_dir}/{population}_rare_association",
        '--threads', str(THREADS)
    ]

    try:
        result = subprocess.run(plink_assoc_cmd, capture_output=True, text=True, check=True)
        print(f"[{datetime.now()}] Association analysis completed for {population}")

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Association analysis failed for {population}")
        print(f"STDERR: {e.stderr}")

    print(f"[{datetime.now()}] Step 3: Preparing for gene-based analysis...")

    bim_rare = f"{output_dir}/{population}_rare_variants.bim"

    if os.path.exists(bim_rare):
        variants_df = pd.read_csv(bim_rare, sep='\t', header=None,
                                names=['CHR', 'SNP', 'CM', 'POS', 'A1', 'A2'])

        print(f"[{datetime.now()}] Rare variants extracted: {len(variants_df):,}")

        variants_df.to_csv(f"{output_dir}/{population}_rare_variant_list.tsv",
                         sep='\t', index=False)

        chr_counts = variants_df['CHR'].value_counts().sort_index()
        print(f"[{datetime.now()}] Variants per chromosome:")
        for chr_num in range(1, 23):
            if chr_num in chr_counts.index:
                print(f"  Chr {chr_num}: {chr_counts[chr_num]:,}")

        save_population_checkpoint(population, 'completed')

        return len(variants_df)

    return 0

def merge_population_results():
    print(f"\n[{datetime.now()}] Merging results from all populations...")

    all_variants = []
    all_associations = []

    for population in POPULATION_PREFIXES:
        variant_file = f"{WORK_DIR}/results/{population}/{population}_rare_variant_list.tsv"

        if os.path.exists(variant_file):
            var_df = pd.read_csv(variant_file, sep='\t')
            var_df['Population'] = population
            all_variants.append(var_df)

        assoc_file = f"{WORK_DIR}/results/{population}/{population}_rare_association.AD.glm.firth"

        if os.path.exists(assoc_file):
            assoc_df = pd.read_csv(assoc_file, sep='\t')
            assoc_df['Population'] = population
            all_associations.append(assoc_df)

    if all_variants:
        combined_variants = pd.concat(all_variants, ignore_index=True)
        print(f"[{datetime.now()}] Total rare variants across all populations: {len(combined_variants):,}")

        combined_variants.to_csv(f"{WORK_DIR}/all_populations_rare_variants.tsv",
                               sep='\t', index=False)

    if all_associations:
        combined_assoc = pd.concat(all_associations, ignore_index=True)

        significant = combined_assoc[combined_assoc['P'] < 0.05]
        print(f"[{datetime.now()}] Significant associations (p < 0.05): {len(significant):,}")

        combined_assoc.to_csv(f"{WORK_DIR}/all_populations_associations.tsv",
                            sep='\t', index=False)

        significant.to_csv(f"{WORK_DIR}/significant_associations.tsv",
                         sep='\t', index=False)

        return combined_variants, combined_assoc

    return None, None

def generate_variant_summary():
    print(f"\n[{datetime.now()}] Generating variant summary report...")

    summary = {
        'Population': [],
        'Total_Samples': [],
        'Rare_Variants': [],
        'Significant_Associations': [],
        'Top_Chr': [],
        'Top_Chr_Count': []
    }

    for population in POPULATION_PREFIXES:
        fam_file = f"{ADSP_DATA_BASE}/{population}.fam"
        if os.path.exists(fam_file):
            fam_df = pd.read_csv(fam_file, sep=' ', header=None)
            n_samples = len(fam_df)
        else:
            n_samples = 0

        variant_file = f"{WORK_DIR}/results/{population}/{population}_rare_variant_list.tsv"
        if os.path.exists(variant_file):
            var_df = pd.read_csv(variant_file, sep='\t')
            n_variants = len(var_df)

            if not var_df.empty:
                chr_counts = var_df['CHR'].value_counts()
                top_chr = chr_counts.index[0]
                top_count = chr_counts.iloc[0]
            else:
                top_chr = 'NA'
                top_count = 0
        else:
            n_variants = 0
            top_chr = 'NA'
            top_count = 0

        assoc_file = f"{WORK_DIR}/results/{population}/{population}_rare_association.AD.glm.firth"
        if os.path.exists(assoc_file):
            assoc_df = pd.read_csv(assoc_file, sep='\t')
            n_sig = len(assoc_df[assoc_df['P'] < 0.05])
        else:
            n_sig = 0

        summary['Population'].append(population)
        summary['Total_Samples'].append(n_samples)
        summary['Rare_Variants'].append(n_variants)
        summary['Significant_Associations'].append(n_sig)
        summary['Top_Chr'].append(top_chr)
        summary['Top_Chr_Count'].append(top_count)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{WORK_DIR}/rare_variant_extraction_summary.tsv",
                     sep='\t', index=False)

    print("\n=== Rare Variant Extraction Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nTotal samples analyzed: {summary_df['Total_Samples'].sum():,}")
    print(f"Total rare variants found: {summary_df['Rare_Variants'].sum():,}")
    print(f"Total significant associations: {summary_df['Significant_Associations'].sum():,}")

    return summary_df

if __name__ == "__main__":
    print("=" * 80)
    print("ADSP Genome-Wide Rare Variant Analysis")
    print("Phase 1B: Extract Rare Variants from 36,000 Samples")
    print("=" * 80)

    if not os.path.exists(f"{WORK_DIR}/matched_samples.tsv"):
        print("ERROR: Please run 01_process_phenotypes.py first")
        sys.exit(1)

    variant_counts = {}

    checkpoint_file = f"{WORK_DIR}/.phase1b_checkpoint.json"
    completed_populations = []

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            for pop, data in checkpoint.get('populations', {}).items():
                if data.get('status') == 'completed':
                    completed_populations.append(pop)
                    print(f"[{datetime.now()}] {pop} already completed (from checkpoint)")

    for population in POPULATION_PREFIXES:
        print(f"\n{'=' * 40}")
        print(f"Processing {population} population")
        print(f"{'=' * 40}")

        count = run_plink_rare_variant_extraction(population)
        if count == -1:
            result_file = f"{WORK_DIR}/results/{population}/{population}_rare_variant_list.tsv"
            if os.path.exists(result_file):
                df = pd.read_csv(result_file, sep='\t')
                count = len(df)

        variant_counts[population] = count

    print(f"\n{'=' * 40}")
    print("Merging population results")
    print(f"{'=' * 40}")

    combined_variants, combined_assoc = merge_population_results()

    summary = generate_variant_summary()

    print(f"\n[{datetime.now()}] Phase 1B completed successfully")
    print("Next step: Run 03_map_variants_to_genes.py")