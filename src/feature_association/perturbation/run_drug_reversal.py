"""
Main script to run drug reversal analysis.

This script performs comprehensive TF activity reversal analysis to identify
rejuvenating drugs based on their ability to reverse age-associated changes.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ciim.src.common import SAVE_DIR, PLOTS_DIR
from ciim.src.feature_association.helper import retrieve_sig_stats
from ciim.src.feature_association.perturbation.drug_reversal import (
    analyze_drug_reversal,
    filter_rejuvenating_drugs,
    save_reversal_results,
    print_reversal_summary
)
from ciim.src.feature_association.perturbation.drug_reversal_plots import (
    plot_reversal_heatmap,
    plot_drug_ranking,
    plot_contingency_tables,
    plot_reversal_overview
)


def run_reversal_analysis(
    dataset: str,
    cell_types: list,
    feature_type: str = 'tf_activity',
    sig_threshold: float = 0.05,
    use_all_drug_stats: bool = True
):
    """
    Run complete drug reversal analysis.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'op', 'CXCL9', 'parsebioscience')
    cell_types : list
        List of cell types to analyze
    feature_type : str
        Feature type ('tf_activity' or 'gene_expression')
    sig_threshold : float
        Significance threshold for individual TF changes
    use_all_drug_stats : bool
        If True, use stats_drugs_all file (all TFs), otherwise use standard file
    
    Returns
    -------
    pd.DataFrame
        Reversal analysis results
    """
    print(f"\n{'='*80}")
    print(f"DRUG REVERSAL ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset}")
    print(f"Feature type: {feature_type}")
    print(f"Cell types: {cell_types}")
    print(f"Significance threshold: {sig_threshold}")
    print(f"Use all drug stats: {use_all_drug_stats}\n")
    
    # Load aging statistics
    print("Loading aging statistics...")
    aging_stats = retrieve_sig_stats(type='bulk', race='both', feature_type=feature_type)
    aging_stats = aging_stats.drop_duplicates(subset=['cell_type', 'tf'])
    print(f"  Loaded {len(aging_stats)} aging TF associations")
    print(f"  Cell types: {aging_stats['cell_type'].unique().tolist()}")
    print(f"  Unique TFs: {aging_stats['tf'].nunique()}\n")
    
    # Load drug statistics
    print("Loading drug statistics...")
    if use_all_drug_stats:
        drug_stats_file = f"{SAVE_DIR}/stats/stats_drugs_all_{dataset}_{feature_type}.csv"
    else:
        drug_stats_file = f"{SAVE_DIR}/stats/stats_drugs_{dataset}_{feature_type}.csv"
    
    if not os.path.exists(drug_stats_file):
        raise FileNotFoundError(
            f"Drug statistics file not found: {drug_stats_file}\n"
            f"Please run compute_all_drug_stats.py first."
        )
    
    drug_stats = pd.read_csv(drug_stats_file)
    print(f"  Loaded {len(drug_stats)} drug TF associations")
    print(f"  Unique drugs/comparisons: {drug_stats['comparision'].nunique()}")
    print(f"  Drugs: {drug_stats['comparision'].unique().tolist()}")
    print(f"  Cell types: {drug_stats['cell_type'].unique().tolist()}")
    print(f"  Unique TFs: {drug_stats['tf'].nunique()}\n")
    
    # Run reversal analysis
    print("Running reversal analysis...")
    results_df = analyze_drug_reversal(
        aging_stats=aging_stats,
        drug_stats=drug_stats,
        cell_types=cell_types,
        dataset=dataset,
        sig_threshold=sig_threshold
    )
    
    if len(results_df) == 0:
        print("No results generated. Check your data.")
        return results_df
    
    print(f"  Analyzed {len(results_df)} drug-cell type combinations\n")
    
    # Save results
    output_path = save_reversal_results(results_df, dataset)
    
    # Print summary
    print_reversal_summary(results_df, dataset)
    
    # Create plots
    print("Generating plots...")
    plot_dir = f"{PLOTS_DIR}/perturbations"
    os.makedirs(plot_dir, exist_ok=True)
    
    try:
        # Reversal heatmap
        print("  Creating reversal heatmap...")
        fig = plot_reversal_heatmap(results_df, dataset, cell_types)
        plt.savefig(f"{plot_dir}/reversal_heatmap_{dataset}.png", 
                   bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        
        # Drug ranking per cell type
        for cell_type in cell_types:
            if cell_type in results_df['cell_type'].values:
                print(f"  Creating drug ranking for {cell_type}...")
                fig = plot_drug_ranking(results_df, cell_type, dataset)
                plt.savefig(f"{plot_dir}/drug_ranking_{dataset}_{cell_type}.png",
                           bbox_inches='tight', dpi=300, transparent=True)
                plt.close()
        
        # Contingency tables for top drugs
        print("  Creating contingency tables...")
        fig = plot_contingency_tables(results_df, dataset, n_top=6)
        plt.savefig(f"{plot_dir}/contingency_tables_{dataset}.png",
                   bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        
        # Overview plot
        print("  Creating overview plot...")
        fig = plot_reversal_overview(results_df, dataset)
        plt.savefig(f"{plot_dir}/overview_{dataset}.png",
                   bbox_inches='tight', dpi=300, transparent=True)
        plt.close()
        
        print(f"\nPlots saved to: {plot_dir}/")
        
    except Exception as e:
        print(f"\nWarning: Error generating plots: {e}")
        print("Results were saved but plots may be incomplete.")
    
    # Filter rejuvenating drugs
    print("\nFiltering for rejuvenating drugs...")
    rejuvenating = filter_rejuvenating_drugs(
        results_df,
        p_threshold=0.05,
        min_reversal_score=0.2,
        min_common_tfs=3
    )
    
    if len(rejuvenating) > 0:
        rejuv_file = f"{SAVE_DIR}/perturbations/rejuvenating_drugs_{dataset}.csv"
        rejuvenating.to_csv(rejuv_file, index=False)
        print(f"Saved {len(rejuvenating)} rejuvenating drugs to: {rejuv_file}")
    else:
        print("No drugs met the rejuvenating criteria.")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return results_df


def main():
    """Main function for command-line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run drug reversal analysis'
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
        '--sig-threshold',
        type=float,
        default=0.05,
        help='Significance threshold for individual TF changes'
    )
    parser.add_argument(
        '--use-standard-stats',
        action='store_true',
        help='Use standard drug stats instead of all-TFs stats'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results_df = run_reversal_analysis(
        dataset=args.dataset,
        cell_types=args.cell_types,
        feature_type=args.feature_type,
        sig_threshold=args.sig_threshold,
        use_all_drug_stats=not args.use_standard_stats
    )
    
    return results_df


if __name__ == '__main__':
    main()
