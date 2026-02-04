import os
import sys
import json
import time
import pickle
import signal
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from alphagenome.data import genome
    from alphagenome.models import dna_client
except ImportError as e:
    print(f"[ERROR] AlphaGenome package not installed: {e}")
    sys.exit(1)

from shared_rate_limiter import SharedRateLimiter


ALL_MODALITIES = [
    dna_client.OutputType.RNA_SEQ,
    dna_client.OutputType.CAGE,
    dna_client.OutputType.PROCAP,
    dna_client.OutputType.SPLICE_SITES,
    dna_client.OutputType.SPLICE_SITE_USAGE,
    dna_client.OutputType.SPLICE_JUNCTIONS,
    dna_client.OutputType.ATAC,
    dna_client.OutputType.DNASE,
    dna_client.OutputType.CHIP_HISTONE,
    dna_client.OutputType.CHIP_TF,
    dna_client.OutputType.CONTACT_MAPS
]

ALL_CELL_TYPES = [
    'UBERON:0000955',
    'CL:0000540',
]

INTERVAL_SIZE = 524288
CHECKPOINT_INTERVAL = 100


def safe_max_effect(values_alt, values_ref, modality_name: str) -> float:
    try:
        if values_alt is None or values_ref is None:
            return 0.0

        alt_array = np.array(values_alt) if hasattr(values_alt, '__iter__') else np.array([values_alt])
        ref_array = np.array(values_ref) if hasattr(values_ref, '__iter__') else np.array([values_ref])

        if alt_array.size == 0 or ref_array.size == 0:
            return 0.0

        diff = np.abs(alt_array - ref_array)
        if diff.size == 0:
            return 0.0

        return float(np.max(diff))

    except Exception as e:
        return 0.0


def create_alphagenome_client():
    try:
        client = dna_client.create(os.environ['ALPHAGENOME_API_KEY'])
        return client
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to create AlphaGenome client: {e}")
        sys.exit(1)


class ParallelWorker:

    def __init__(self, proc_id, chunk_file, rate_limiter_state_file='rate_limiter_state.json'):
        self.proc_id = proc_id
        self.chunk_file = chunk_file
        self.checkpoint_file = f'worker_checkpoints/checkpoint_{proc_id:03d}.json'
        self.results_file = f'worker_results/results_{proc_id:03d}.pkl'

        Path('worker_checkpoints').mkdir(exist_ok=True)
        Path('worker_results').mkdir(exist_ok=True)

        self.rate_limiter = SharedRateLimiter(
            state_file=rate_limiter_state_file,
            hourly_limit=3600,
            daily_limit=50000,
            min_delay=1.0
        )

        print(f"[{datetime.now()}] Worker {proc_id}: Initializing AlphaGenome client...")
        self.model = create_alphagenome_client()
        print(f"[{datetime.now()}] Worker {proc_id}: Client ready")

        self.processed_variants = set()
        self.results = []
        self.load_checkpoint()

        signal.signal(signal.SIGTERM, self.save_and_exit)
        signal.signal(signal.SIGINT, self.save_and_exit)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                cp = json.load(f)
                self.processed_variants = set(cp.get('processed', []))
                print(f"[{datetime.now()}] Worker {self.proc_id}: Loaded {len(self.processed_variants)} processed variants")

        if os.path.exists(self.results_file):
            with open(self.results_file, 'rb') as f:
                self.results = pickle.load(f)

    def save_checkpoint(self):
        checkpoint_data = {
            'proc_id': self.proc_id,
            'processed': list(self.processed_variants),
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.processed_variants)
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        with open(self.results_file, 'wb') as f:
            pickle.dump(self.results, f)

    def save_and_exit(self, signum, frame):
        print(f"\n[{datetime.now()}] Worker {self.proc_id}: Saving checkpoint before exit...")
        self.save_checkpoint()
        sys.exit(0)

    def process_variant(self, variant):
        if not self.rate_limiter.acquire(timeout=3600):
            print(f"[{datetime.now()}] Worker {self.proc_id}: Rate limiter timeout!")
            return None

        try:
            chr_str = f"chr{variant['chr']}"
            position = int(variant['pos'])
            ref_allele = variant['ref']
            alt_allele = variant['alt']

            interval_start = max(0, position - INTERVAL_SIZE // 2)
            interval_end = interval_start + INTERVAL_SIZE
            interval = genome.Interval(chr_str, interval_start, interval_end)

            variant_obj = genome.Variant(
                chromosome=chr_str,
                position=position,
                reference_bases=ref_allele,
                alternate_bases=alt_allele
            )

            result = {
                'variant_id': variant['variant_id'],
                'gene_name': variant['gene_name'],
                'gene_id': variant['gene_id'],
                'rna_seq_effect': 0.0,
                'cage_effect': 0.0,
                'procap_effect': 0.0,
                'splice_sites_effect': 0.0,
                'splice_site_usage_effect': 0.0,
                'splice_junctions_effect': 0.0,
                'atac_effect': 0.0,
                'dnase_effect': 0.0,
                'chip_histone_effect': 0.0,
                'chip_tf_effect': 0.0,
                'contact_maps_effect': 0.0
            }

            for cell_type in ALL_CELL_TYPES:
                try:
                    outputs = self.model.predict_variant(
                        interval=interval,
                        variant=variant_obj,
                        ontology_terms=[cell_type],
                        requested_outputs=ALL_MODALITIES
                    )

                    if outputs:
                        if outputs.alternate.rna_seq and outputs.reference.rna_seq:
                            effect = safe_max_effect(outputs.alternate.rna_seq.values,
                                                    outputs.reference.rna_seq.values, 'RNA_SEQ')
                            result['rna_seq_effect'] = max(result['rna_seq_effect'], effect)

                        if outputs.alternate.cage and outputs.reference.cage:
                            effect = safe_max_effect(outputs.alternate.cage.values,
                                                    outputs.reference.cage.values, 'CAGE')
                            result['cage_effect'] = max(result['cage_effect'], effect)

                        if outputs.alternate.procap and outputs.reference.procap:
                            effect = safe_max_effect(outputs.alternate.procap.values,
                                                    outputs.reference.procap.values, 'PROCAP')
                            result['procap_effect'] = max(result['procap_effect'], effect)

                        if outputs.alternate.splice_sites and outputs.reference.splice_sites:
                            effect = safe_max_effect(outputs.alternate.splice_sites.values,
                                                    outputs.reference.splice_sites.values, 'SPLICE_SITES')
                            result['splice_sites_effect'] = max(result['splice_sites_effect'], effect)

                        if outputs.alternate.splice_site_usage and outputs.reference.splice_site_usage:
                            effect = safe_max_effect(outputs.alternate.splice_site_usage.values,
                                                    outputs.reference.splice_site_usage.values, 'SPLICE_SITE_USAGE')
                            result['splice_site_usage_effect'] = max(result['splice_site_usage_effect'], effect)

                        if outputs.alternate.splice_junctions and outputs.reference.splice_junctions:
                            effect = safe_max_effect(outputs.alternate.splice_junctions.values,
                                                    outputs.reference.splice_junctions.values, 'SPLICE_JUNCTIONS')
                            result['splice_junctions_effect'] = max(result['splice_junctions_effect'], effect)

                        if outputs.alternate.atac and outputs.reference.atac:
                            effect = safe_max_effect(outputs.alternate.atac.values,
                                                    outputs.reference.atac.values, 'ATAC')
                            result['atac_effect'] = max(result['atac_effect'], effect)

                        if outputs.alternate.dnase and outputs.reference.dnase:
                            effect = safe_max_effect(outputs.alternate.dnase.values,
                                                    outputs.reference.dnase.values, 'DNASE')
                            result['dnase_effect'] = max(result['dnase_effect'], effect)

                        if outputs.alternate.chip_histone and outputs.reference.chip_histone:
                            effect = safe_max_effect(outputs.alternate.chip_histone.values,
                                                    outputs.reference.chip_histone.values, 'CHIP_HISTONE')
                            result['chip_histone_effect'] = max(result['chip_histone_effect'], effect)

                        if outputs.alternate.chip_tf and outputs.reference.chip_tf:
                            effect = safe_max_effect(outputs.alternate.chip_tf.values,
                                                    outputs.reference.chip_tf.values, 'CHIP_TF')
                            result['chip_tf_effect'] = max(result['chip_tf_effect'], effect)

                        if outputs.alternate.contact_maps and outputs.reference.contact_maps:
                            effect = safe_max_effect(outputs.alternate.contact_maps.values,
                                                    outputs.reference.contact_maps.values, 'CONTACT_MAPS')
                            result['contact_maps_effect'] = max(result['contact_maps_effect'], effect)

                    time.sleep(0.1)

                except Exception as cell_e:
                    print(f"[{datetime.now()}] Worker {self.proc_id}: Cell type {cell_type} failed for {variant['variant_id']}: {cell_e}")
                    continue

            return result

        except Exception as e:
            print(f"[{datetime.now()}] Worker {self.proc_id}: ERROR processing {variant['variant_id']}: {e}")
            return None

    def process_all(self):
        variants_df = pd.read_csv(self.chunk_file, sep='\t')
        total_variants = len(variants_df)

        print(f"[{datetime.now()}] Worker {self.proc_id}: Processing {total_variants:,} variants")
        print(f"[{datetime.now()}] Worker {self.proc_id}: Already processed: {len(self.processed_variants):,}")

        start_time = datetime.now()
        processed_count = len(self.processed_variants)

        for idx, variant in variants_df.iterrows():
            if variant['variant_id'] in self.processed_variants:
                continue

            result = self.process_variant(variant)

            if result:
                self.processed_variants.add(variant['variant_id'])
                self.results.append(result)
                processed_count += 1

                if processed_count % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = processed_count / elapsed * 60 if elapsed > 0 else 0
                    pct = 100 * processed_count / total_variants
                    stats = self.rate_limiter.get_stats()
                    print(f"[{datetime.now()}] Worker {self.proc_id}: {processed_count:,}/{total_variants:,} "
                          f"({pct:.1f}%) | {rate:.2f} var/min | "
                          f"Daily: {stats['daily_calls']:,}/{stats['daily_limit']:,}")

                if processed_count % CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint()

        self.save_checkpoint()

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[{datetime.now()}] Worker {self.proc_id}: COMPLETED {processed_count:,} variants in {elapsed/60:.1f} minutes")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 parallel_alphgenome_worker.py <proc_id> <chunk_file>")
        sys.exit(1)

    proc_id = int(sys.argv[1])
    chunk_file = sys.argv[2]

    print(f"[{datetime.now()}] Starting Worker {proc_id} with chunk {chunk_file}")

    worker = ParallelWorker(proc_id, chunk_file)
    worker.process_all()

    print(f"[{datetime.now()}] Worker {proc_id} finished successfully")
