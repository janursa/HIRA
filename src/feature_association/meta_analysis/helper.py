

import pandas as pd
import subprocess
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))

def check_signs(group):
    '''
    slope should be same for both datasets
    '''
    signs = group['slope'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return signs.nunique() == 1

def run_meta_analysis(stats_all, meta_analysis_type='max', min_degree=2, temp_dir='../output/tf_activity/'):
    # ---------- prepare
    
    assert stats_all.shape[0]> 0, 'No stats for meta analysis'
    
    print('Meta analysis...')
    stats_all_c = stats_all.copy()
    stats_all_c.rename(columns={'p_value': 'pvalue'}, inplace=True)
    
    original_name = 'gene'
    if 'tf' in stats_all_c.columns:
        stats_all_c.rename(columns={'tf': 'gene'}, inplace=True)
        original_name = 'tf'
    elif 'target' in stats_all_c.columns:
        stats_all_c.rename(columns={'target': 'gene'}, inplace=True)
        original_name = 'target'
    elif 'pathway' in stats_all_c.columns:
        stats_all_c.rename(columns={'pathway': 'gene'}, inplace=True)
        original_name = 'pathway'
    else:
        raise ValueError('No gene, tf or target column in stats_all')
    # -------- actual run
    if min_degree is not None:
        stats_all_c = stats_all_c.groupby(['gene', 'cell_type']).filter(lambda group: group['dataset'].nunique() >= min_degree)
    
    cell_types = stats_all_c['cell_type'].unique()
    df_meta_store = []
    for cell_type in cell_types:
        df = stats_all_c[stats_all_c['cell_type'] == cell_type]
        df['pvalue'] = df['pvalue']+1E-20 # to avoid 0 p value
        
        file_path = f'{temp_dir}/stats_{cell_type}.csv'
        df.to_csv(file_path, index=False)
        out_path = f'{temp_dir}/stats_{cell_type}_meta.csv'

        Rscript_file = f'{current_dir}/meta_analysis.R'
        # try:
        # Run the R script with the provided file paths
        try:
            result = subprocess.run(
                ["Rscript", Rscript_file, file_path, out_path, meta_analysis_type],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while running R script: {Rscript_file}")
            print("STDOUT:", e.stdout.decode())  # Standard Output
            print("STDERR:", e.stderr.decode())  # Error Output from R
            raise
        df_meta = pd.read_csv(out_path)
        assert df_meta.isna().sum().sum() == 0
        df_meta['cell_type'] = cell_type
        print(cell_type, df_meta.shape)
        df_meta_store.append(df_meta)
    if df_meta_store:
        df_meta_all = pd.concat(df_meta_store)
    else:
        df_meta_all = pd.DataFrame() 
    
    
    df_meta_all.reset_index(drop=True)

    df_meta_all.rename(columns={'gene': original_name}, inplace=True)
    if df_meta_all.shape[0]==0:
        print(f"No meta analysis results for {stats_all['cell_type'].unique()}")
        return None
    stats_all = stats_all.merge(df_meta_all, on=[original_name, 'cell_type'], how='left')
    return stats_all
