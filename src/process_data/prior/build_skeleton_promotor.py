"""
Build the promoter-based TF->gene skeleton (ENCODE + JASPAR motifs vs TSS flanks).

Same recipe as task_grn_inference/src/process_data/skeleton/skeleton_motif_only.py
(and notebooks/prepare_resources.ipynb), reimplemented with bedtools instead of
scglue, since no scglue install/image is available here.

Usage:
    python src/process_data/prior/build_skeleton_promotor.py
"""
import argparse
import os
import shutil
import subprocess
import sys
import pandas as pd

sys.path.insert(0, 'src')
from config import PRIOR_DIR

TASK_GRN_REPO = os.environ['TASK_GRN_BENCHMARK_DIR']
TSS_BED = f'{TASK_GRN_REPO}/resources/supp_data/tss_h38.bed'
MOTIF_FILES = {
    'encode': f'{TASK_GRN_REPO}/resources/supp_data/databases/scglue/ENCODE-TF-ChIP-hg38.bed.gz',
    'jaspar': f'{TASK_GRN_REPO}/resources/supp_data/databases/scglue/JASPAR2022-hg38.bed.gz',
}
TF_ALL = f'{PRIOR_DIR}/tf_all.csv'
BEDTOOLS = shutil.which('bedtools') or 'bedtools'

parser = argparse.ArgumentParser()
parser.add_argument('--flank_length', type=int, default=1000)
parser.add_argument('--out', default=f'{PRIOR_DIR}/skeleton_promotor.csv')
parser.add_argument('--temp_dir', default='temp/skeleton_promotor')
args = parser.parse_args()

os.makedirs(args.temp_dir, exist_ok=True)


def sh(cmd):
    print('+', cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. Promoter windows: TSS +/- flank_length, from tss_h38.bed
tss = pd.read_csv(TSS_BED, sep='\t', comment='#',
                   names=['chrom', 'start', 'end', 'gene', 'score', 'strand', 'transcript_type'])
tss = tss[tss['gene'].notna() & (tss['gene'] != '')]
tss['start'] = (tss['start'] - args.flank_length).clip(lower=0)
tss['end'] = tss['end'] + args.flank_length

windows_bed = f'{args.temp_dir}/promoter_windows.bed'
tss[['chrom', 'start', 'end', 'gene']].to_csv(windows_bed, sep='\t', header=False, index=False)

windows_sorted = f'{args.temp_dir}/promoter_windows.sorted.bed'
sh(f'sort -k1,1 -k2,2n {windows_bed} > {windows_sorted}')

# 2. Known TFs (filter sources)
tf_set = set(pd.read_csv(TF_ALL, header=None, names=['TF'])['TF'])

# 3. Intersect promoter windows with each motif file -> TF->gene edges
parts = []
for label, motif_gz in MOTIF_FILES.items():
    motif_sorted = f'{args.temp_dir}/motif_{label}.sorted.bed'
    sh(f'zcat {motif_gz} | sort -k1,1 -k2,2n > {motif_sorted}')

    overlap_file = f'{args.temp_dir}/overlap_{label}.tsv'
    sh(f'{BEDTOOLS} intersect -sorted -wa -wb -a {windows_sorted} -b {motif_sorted} > {overlap_file}')

    df = pd.read_csv(overlap_file, sep='\t', header=None,
                      names=['w_chrom', 'w_start', 'w_end', 'target',
                             'm_chrom', 'm_start', 'm_end', 'source'])
    df = df[['source', 'target']]
    df = df[df['source'].isin(tf_set)]
    print(f'{label}: {len(df)} TF-gene edges', flush=True)
    parts.append(df)

skeleton = pd.concat(parts).drop_duplicates()
skeleton['edge'] = skeleton['source'] + '_' + skeleton['target']
skeleton = skeleton[['source', 'target', 'edge']].reset_index(drop=True)

print(f'Final skeleton: {len(skeleton)} edges, '
      f'{skeleton.source.nunique()} TFs, {skeleton.target.nunique()} genes', flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
skeleton.to_csv(args.out, index=False)
print(f'Saved: {args.out}', flush=True)
