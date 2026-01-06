"""
Compute drug association statistics for ALL TFs (not just significant ones).

This script computes TF activity association with drug treatment for all TFs,
enabling comprehensive reversal analysis independent of individual TF significance.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from ciim.src.common import SAVE_DIR, cell_types
from ciim.src.feature_association.helper import (
    wrapper_tf_activity,
    wrapper_gene_expression,
    wrapper_association_with_age_condition
)
from ongoing.ciim.src.config import get_config


def compute_all_drug_stats(
    dataset: str,
    target_celltypes: list,
    feature_type: str = 'tf_activity',
    force_recalculate: bool = False
):
    """
    Compute drug statistics for all TFs in the dataset.
    
    This differs from the standard analysis by NOT filtering for significant TFs,
    allowing comprehensive reversal analysis.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'op', 'CXCL9', 'parsebioscience')
    target_celltypes : list
        List of cell types to analyze
    feature_type : str
        Feature type ('tf_activity' or 'gene_expression')
    force_recalculate : bool
        If True, recalculate even if output file exists
    
    Returns
    -------
    str
        Path to output CSV file
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING ALL DRUG STATS - {dataset.upper()}")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset}")
    print(f"Feature type: {feature_type}")
    print(f"Cell types: {target_celltypes}")
    
    # Output file
    output_file = f"{SAVE_DIR}/stats/stats_drugs_all_{dataset}_{feature_type}.csv"
    
    # Check if already exists
    if os.path.exists(output_file) and not force_recalculate:
        print(f"\nOutput file already exists: {output_file}")
        print("Use force_recalculate=True to recompute.")
        stats_drugs = pd.read_csv(output_file)
        print(f"\nLoaded {len(stats_drugs)} rows from existing file.")
        return output_file
    
    # Get dataset configuration
    cfg = get_config(dataset)
    
    # Determine data type
    if feature_type == 'tf_activity':
        data_type = 'sc'
    else:
        data_type = 'bulk'
    
    # Override for parsebioscience and op
    if dataset in ["parsebioscience", "op"]:
        data_type = "bulk"
    
    print(f"Data type: {data_type}")
    print(f"Test type: {cfg['test_type']}\n")
    
    # Process each cell type
    stats_drugs_store = []
    
    for cell_type in tqdm(target_celltypes, desc='Processing cell types'):
        print(f"\n  Processing {cell_type}...")
        
        # Set up parameters
        par = {
            "feature_type": feature_type,
            "datasets": [dataset],
            "cell_types": [cell_type],
            "type": data_type,
            "cell_type_resolution": "cell_type",
            "association_type": "spearman",
        }
        
        # Calculate TF activity or gene expression
        try:
            if feature_type == "tf_activity":
                print(f"    Calculating TF activity...")
                wrapper_tf_activity(par)
            elif feature_type == "gene_expression":
                print(f"    Calculating gene expression...")
                wrapper_gene_expression(par)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
        except Exception as e:
            print(f"    WARNING: Error calculating {feature_type} for {cell_type}: {e}")
            continue
        
        # Run association analysis for ALL features (not filtering)
        try:
            print(f"    Running association with treatment...")
            stats_drugs = wrapper_association_with_age_condition(
                par,
                features=None,  # ALL features
                test_type=cfg["test_type"]
            )
            
            # Add comparison names
            if cfg["name_mapping"] == "all":
                # For "all" mode, keep all conditions except control as comparisons
                stats_drugs["comparision"] = stats_drugs["condition"]
                # Filter out control if specified
                if "control_name" in cfg:
                    stats_drugs = stats_drugs[stats_drugs["comparision"] != cfg["control_name"]]
            else:
                stats_drugs["comparision"] = stats_drugs["condition"].replace(cfg["name_mapping"])
                stats_drugs = stats_drugs.dropna(subset=["comparision"])
            
            stats_drugs["comparision"] = stats_drugs["comparision"].astype("category")
            
            # Add cell type
            stats_drugs['cell_type'] = cell_type
            
            print(f"    Computed stats for {len(stats_drugs)} TFs")
            stats_drugs_store.append(stats_drugs)
            
        except Exception as e:
            print(f"    WARNING: Error in association analysis for {cell_type}: {e}")
            continue
    
    if len(stats_drugs_store) == 0:
        raise ValueError("No statistics computed for any cell type!")
    
    # Combine all results
    print(f"\nCombining results from {len(stats_drugs_store)} cell types...")
    stats_drugs_all = pd.concat(stats_drugs_store, ignore_index=True)
    
    # Save to CSV
    os.makedirs(f"{SAVE_DIR}/stats", exist_ok=True)
    stats_drugs_all.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY:")
    print(f"{'='*80}")
    print(f"Total rows: {len(stats_drugs_all)}")
    print(f"Unique TFs: {stats_drugs_all['tf'].nunique()}")
    print(f"Unique comparisons: {stats_drugs_all['comparision'].nunique()}")
    print(f"Cell types: {stats_drugs_all['cell_type'].unique().tolist()}")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*80}\n")
    
    return output_file


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compute drug statistics for all TFs'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='op',
        choices=['op', 'CXCL9', 'parsebioscience'],
        help='Dataset name'
    )
    parser.add_argument(
        '--cell-types',
        type=str,
        nargs='+',
        default=['CD4T', 'CD8T'],
        help='Cell types to analyze'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='tf_activity',
        choices=['tf_activity', 'gene_expression'],
        help='Feature type'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recalculation even if output exists'
    )
    
    args = parser.parse_args()
    
    # Run computation
    compute_all_drug_stats(
        dataset=args.dataset,
        target_celltypes=args.cell_types,
        feature_type=args.feature_type,
        force_recalculate=args.force
    )


if __name__ == '__main__':
    main()
