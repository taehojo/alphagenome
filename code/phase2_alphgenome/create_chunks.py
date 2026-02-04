import os
import json
import pandas as pd
from pathlib import Path

def create_chunks(num_processes, test_mode=False, variants_per_test=100):
    variants_df = pd.read_csv('variants_for_alphgenome.tsv', sep='\t')
    total_variants = len(variants_df)
    print(f"Total variants: {total_variants:,}")

    processed_variants = set()
    if os.path.exists('alphgenome_checkpoint.json'):
        with open('alphgenome_checkpoint.json', 'r') as f:
            cp = json.load(f)
            processed_variants = set(cp.get('processed', []))

    print(f"Already processed: {len(processed_variants):,}")

    remaining_df = variants_df[~variants_df['variant_id'].isin(processed_variants)]
    remaining_count = len(remaining_df)
    print(f"Remaining to process: {remaining_count:,}")

    if remaining_count == 0:
        print("No variants remaining to process!")
        return []

    if test_mode:
        test_total = num_processes * variants_per_test
        remaining_df = remaining_df.head(test_total)
        remaining_count = len(remaining_df)
        print(f"Test mode: Using {remaining_count:,} variants ({variants_per_test} per process)")

    chunk_size = remaining_count // num_processes
    remainder = remaining_count % num_processes

    chunks_dir = Path('variant_chunks')
    chunks_dir.mkdir(exist_ok=True)

    chunk_info = []
    start_idx = 0

    for proc_id in range(num_processes):
        current_chunk_size = chunk_size + (1 if proc_id < remainder else 0)

        if current_chunk_size == 0:
            break

        end_idx = start_idx + current_chunk_size
        chunk_df = remaining_df.iloc[start_idx:end_idx]

        chunk_file = chunks_dir / f'chunk_{proc_id:03d}.tsv'
        chunk_df.to_csv(chunk_file, sep='\t', index=False)

        chunk_info.append({
            'proc_id': proc_id,
            'chunk_file': str(chunk_file),
            'variant_count': len(chunk_df),
            'start_idx': start_idx,
            'end_idx': end_idx
        })

        print(f"  Chunk {proc_id}: {len(chunk_df):,} variants -> {chunk_file}")

        start_idx = end_idx

    metadata = {
        'num_processes': num_processes,
        'test_mode': test_mode,
        'total_variants': total_variants,
        'already_processed': len(processed_variants),
        'remaining_variants': remaining_count,
        'chunks': chunk_info
    }

    with open('variant_chunks/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCreated {len(chunk_info)} chunks in {chunks_dir}/")
    print(f"Metadata saved to {chunks_dir}/metadata.json")

    return chunk_info


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 create_chunks.py <num_processes> [--test]")
        print("\nExamples:")
        print("  python3 create_chunks.py 3 --test    # Test mode: 3 processes, 100 variants each")
        print("  python3 create_chunks.py 32           # Full mode: 32 processes, all remaining variants")
        sys.exit(1)

    num_processes = int(sys.argv[1])
    test_mode = '--test' in sys.argv

    print(f"Creating chunks for {num_processes} processes...")
    if test_mode:
        print("TEST MODE: Small subset only\n")
    else:
        print("FULL MODE: All remaining variants\n")

    chunks = create_chunks(num_processes, test_mode=test_mode)

    if chunks:
        print(f"\n✓ Successfully created {len(chunks)} chunks")
    else:
        print("\n✗ No chunks created")
