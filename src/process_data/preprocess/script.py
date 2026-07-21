
import anndata as ad
import gc
from hira.src.config import DATASET_NAME_MAPPING, get_config, MAJOR_CT_LABEL, PRIOR_DIR
import argparse
import glob
import os
import numpy as np
from hira.src.process_data.preprocess.helper import annotate_celltypes, basic_qc, format_data

def subset_to_test(adata):
    print('Test mode: subsetting data', flush=True)
    cell_indices = adata.obs.groupby('bulk_group').apply(lambda x: x.index[0]).values
    adata = adata[cell_indices, :].to_memory()
    return adata

def load_sc_data(file_name, dataset, run_test, gene_names=None):
    if dataset == 'soundlife':
        print(f'Soundlife: Processing multiple files from directory: {file_name}', flush=True)
        input_pattern = os.path.join(file_name, 'SoundLife_*.h5ad')
        input_files = sorted(glob.glob(input_pattern))
        if not input_files:
            raise ValueError(f"No SoundLife_*.h5ad files found in {file_name}")
        print(f'Found {len(input_files)} SoundLife files to merge:', flush=True)
        for f in input_files:
            print(f'  - {os.path.basename(f)}', flush=True)
        merged_adata = None
        if run_test:
            input_files = input_files[:2]

        for i, input_file in enumerate(input_files):
            print(f'\nReading file {i+1}/{len(input_files)}: {os.path.basename(input_file)}', flush=True)
            adata_temp = ad.read_h5ad(input_file, backed='r')
            adata_temp = format_data(adata_temp, dataset)
            if run_test:
                adata_temp = subset_to_test(adata_temp)
            elif gene_names is not None:
                # Filter to known genes before loading into memory (saves ~50% memory)
                keep = adata_temp.var_names.isin(gene_names)
                adata_temp = adata_temp[:, keep].to_memory()
            else:
                adata_temp = adata_temp.to_memory()

            if merged_adata is None:
                merged_adata = adata_temp
            else:
                merged_adata = ad.concat(
                    [merged_adata, adata_temp],
                    join='inner',
                    merge='same'
                )
            print(f'Merged shape so far: {merged_adata.shape}', flush=True)
            del adata_temp
            gc.collect()

        adata = merged_adata
        print(f'\nFinal merged soundlife shape: {adata.shape}', flush=True)

    else:
        adata = ad.read_h5ad(file_name, backed='r')
        adata = format_data(adata, dataset)
        if run_test:
            adata = subset_to_test(adata)
            print(f'Kept {adata.shape[0]} cells from groups', flush=True)
            if dataset == 'op' and 'counts' in adata.layers:
                import scipy.sparse as sp
                raw = adata.layers['counts']
                raw = raw.tocsr() if sp.issparse(raw) else sp.csr_matrix(raw)
                adata.X = raw.astype(np.int32)
                del adata.layers['counts']
        # Non-test: return backed — main() will chunk by bulk_group + gene filter
        # in a single combined mask to avoid chained masks on backed objects.
    return adata


def _build_chunk_groups(group_sizes, chunk_max):
    """Accumulate bulk_group keys into chunks of at most chunk_max cells."""
    chunks, current_groups, current_size = [], [], 0
    for group, size in group_sizes.items():
        if current_groups and current_size + size > chunk_max:
            chunks.append(current_groups)
            current_groups, current_size = [group], size
        else:
            current_groups.append(group)
            current_size += size
    if current_groups:
        chunks.append(current_groups)
    return chunks


def main(par):
    dataset = par['dataset']
    dataset = DATASET_NAME_MAPPING.get(dataset, dataset)
    par['dataset'] = dataset
    file_name = par['input_file']

    gene_names = np.loadtxt(f'{PRIOR_DIR}/gene_names.txt', dtype=str)
    adata = load_sc_data(file_name, dataset, par['run_test'], gene_names=gene_names)

    if dataset == 'soundlife' or par['run_test']:
        adata = basic_qc(adata, par['run_test'])
        if dataset == 'soundlife':
            adata = adata[:, adata.var_names.isin(gene_names)].copy()
        adata = annotate_celltypes(adata, dataset)
    else:
        # Chunked loading: combine cell + gene masks into one backed slice per chunk
        # to avoid chained masking on backed objects (only one mask allowed).
        chunk_max = 500_000
        gene_mask = adata.var_names.isin(gene_names)
        chunk_groups_list = _build_chunk_groups(
            adata.obs['bulk_group'].value_counts(), chunk_max
        )
        print(f'Chunked loading: {len(chunk_groups_list)} chunk(s), max {chunk_max:,} cells each', flush=True)

        processed = []
        qc_reports = []
        for i, chunk_groups in enumerate(chunk_groups_list):
            cell_mask = adata.obs['bulk_group'].isin(chunk_groups)
            chunk = adata[cell_mask, gene_mask].to_memory()
            print(f'Chunk {i+1}/{len(chunk_groups_list)}: {chunk.n_obs:,} cells loaded to memory', flush=True)
            if dataset == 'op' and 'counts' in chunk.layers:
                import scipy.sparse as sp
                raw = chunk.layers['counts']
                raw = raw.tocsr() if sp.issparse(raw) else sp.csr_matrix(raw)
                chunk.X = raw.astype(np.int32)
                del chunk.layers['counts']
            chunk = basic_qc(chunk, par['run_test'])
            chunk = annotate_celltypes(chunk, dataset)
            if 'annotation_qc' in chunk.uns:
                qc_reports.append(chunk.uns.pop('annotation_qc'))
            processed.append(chunk)
            del chunk
            gc.collect()

        adata = ad.concat(processed, join='inner', merge='same')
        del processed
        gc.collect()
        if qc_reports:
            import pandas as pd
            combined_qc = pd.concat([pd.DataFrame(r) for r in qc_reports], ignore_index=True)
            combined_qc.index = combined_qc.index.astype(str)
            adata.uns['annotation_qc'] = combined_qc.to_dict()

    print(f"Writing processed data to {par['processed_files_dir']}/{dataset}.h5ad", flush=True)
    assert not adata.layers, "adata should not have any layers before writing"
    assert not isinstance(adata.X, np.ndarray), "adata.X should be sparse before writing"
    assert np.all(adata.X.data >= 0), "adata.X should contain raw counts (non-negative values)"
    one_value = adata.X.data[0]
    assert np.isclose(one_value % 1, 0), "adata.X should contain raw counts (integer values)"

    adata.write_h5ad(f"{par['processed_files_dir']}/{dataset}.h5ad", compression='gzip')
    print(adata, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_files_dir', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--run-test', action='store_true')
    parser.add_argument('--dataset', required=True)
    par = vars(parser.parse_args())
    main(par)
