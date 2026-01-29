
def verify_dpt_correlation(adata_combined, dataset, cell_type, output_dir):
    """
    Verify that DPT pseudotime correlates with naive-effector marker scores.
    Produces donor-based analysis and overall correlation plot.
    """
    print("\n=== Verifying DPT Correlation with Marker Scores ===")
    corr_output = f"{output_dir}/dpt_marker_correlation_{dataset}_{cell_type}.csv"
    if False:
        # Calculate correlations per donor
        donor_correlations = []
        unique_donors = adata_combined.obs['donor_age'].unique()
        
        for donor_id in unique_donors:
            adata_donor = adata_combined[adata_combined.obs['donor_age'] == donor_id]
            
            # Correlate pseudotime with naive score (should be negative)
            corr_naive, pval_naive = spearmanr(adata_donor.obs['paga_pseudotime'], 
                                            adata_donor.obs['naive_score'])
            
            # Correlate pseudotime with effector score (should be positive)
            corr_effector, pval_effector = spearmanr(adata_donor.obs['paga_pseudotime'], 
                                                    adata_donor.obs['effector_memory_score'])
            
            age = adata_donor.obs['age'].iloc[0]
            
            donor_correlations.append({
                'donor_age': donor_id,
                'age': age,
                'n_cells': adata_donor.n_obs,
                'naive_corr': corr_naive,
                'naive_pval': pval_naive,
                'effector_corr': corr_effector,
                'effector_pval': pval_effector
            })
        
        corr_df = pd.DataFrame(donor_correlations)
        
        corr_df.to_csv(corr_output, index=False)
    else:
        corr_df = pd.read_csv(corr_output)
    print(f"Donor correlations saved to: {corr_output}")
    
    # Print summary
    print(f"\nCorrelation Summary:")
    print(f"Naive score - Mean corr: {corr_df['naive_corr'].mean():.3f} (Expected: negative)")
    print(f"Effector score - Mean corr: {corr_df['effector_corr'].mean():.3f} (Expected: positive)")
    print(f"Significant donors (p<0.05) for naive: {(corr_df['naive_pval'] < 0.05).sum()}/{len(corr_df)}")
    print(f"Significant donors (p<0.05) for effector: {(corr_df['effector_pval'] < 0.05).sum()}/{len(corr_df)}")
    if False:
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Scatter plot of pseudotime vs naive score
        ax = axes[0]
        ax.scatter(adata_combined.obs['paga_pseudotime'], 
                adata_combined.obs['naive_score'],
                alpha=0.01, s=1, c='blue')
        ax.set_xlabel('DPT Pseudotime', fontsize=12)
        ax.set_ylabel('Naive Score', fontsize=12)
        ax.set_title(f'Pseudotime vs Naive Score\nMean ρ={corr_df["naive_corr"].mean():.3f}', fontsize=12)
        
        # Plot 2: Scatter plot of pseudotime vs effector score
        ax = axes[1]
        ax.scatter(adata_combined.obs['paga_pseudotime'], 
                adata_combined.obs['effector_memory_score'],
                alpha=0.01, s=1, c='red')
        ax.set_xlabel('DPT Pseudotime', fontsize=12)
        ax.set_ylabel('Effector Memory Score', fontsize=12)
        ax.set_title(f'Pseudotime vs Effector Score\nMean ρ={corr_df["effector_corr"].mean():.3f}', fontsize=12)
        
        plt.tight_layout()
        plot_output = f"{PLOTS_DIR}/dpt_marker_correlation_{dataset}_{cell_type}.png"
        plt.savefig(plot_output, dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved to: {plot_output}")
        plt.close()

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"MODE: Verify DPT Correlation")
    print(f"{'='*60}\n")
    adata_combined = process_donors(args.dataset, args.cell_type, test_mode=args.test)
    corr_df = verify_dpt_correlation(adata_combined, args.dataset, args.cell_type, OUTPUT_DIR)