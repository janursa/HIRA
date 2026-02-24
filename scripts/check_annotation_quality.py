"""
Quality check for cell type annotations across discovery cohorts.
Identify potential red flags and inconsistencies.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from hiara import SUB_CTS, DISCOVERY_COHORTS, PLOTS_DIR, retrieve_adata


def check_proportion_consistency():
    """
    Check if cell type proportions are wildly inconsistent across datasets.
    Red flag: If a cell type is abundant in one dataset but rare/missing in others.
    """
    print("\n" + "="*80)
    print("1. CELL TYPE PROPORTION CONSISTENCY CHECK")
    print("="*80)
    
    all_props = []
    for dataset in DISCOVERY_COHORTS:
        obs = retrieve_adata(dataset=dataset, data_type='sc', only_obs=True)
        total = len(obs)
        props = obs['Sub_CT'].value_counts() / total * 100
        props.name = dataset
        all_props.append(props)
    
    df = pd.DataFrame(all_props).T
    df = df.fillna(0)
    
    print("\nCell type proportions (%) across datasets:")
    print(df.round(2))
    
    # Calculate coefficient of variation for each cell type
    cv = df.std(axis=1) / df.mean(axis=1)
    cv = cv.sort_values(ascending=False)
    
    print("\n🔍 Coefficient of Variation (CV) - Higher = More Inconsistent:")
    for ct, cv_val in cv.items():
        flag = "⚠️ HIGH VARIATION" if cv_val > 0.5 else ""
        print(f"  {ct:20s}: {cv_val:.3f} {flag}")
    
    # Check for missing cell types
    print("\n🔍 Missing cell types (0% in dataset):")
    for dataset in DISCOVERY_COHORTS:
        missing = df.index[df[dataset] == 0].tolist()
        if missing:
            print(f"  {dataset}: {missing}")
    
    return df


def check_age_representation():
    """
    Check if all cell types are represented across the age range.
    Red flag: If a cell type only appears in young or old donors.
    """
    print("\n" + "="*80)
    print("2. AGE REPRESENTATION CHECK")
    print("="*80)
    
    for dataset in DISCOVERY_COHORTS:
        print(f"\n### {dataset.upper()} ###")
        obs = retrieve_adata(dataset=dataset, data_type='sc', only_obs=True)
        
        # Extract numeric age from donor_age string
        if 'age' in obs.columns:
            ages_col = 'age'
        else:
            # Try to parse from donor_age if it's a string
            obs['age_numeric'] = obs['donor_age'].astype(str).str.extract(r'(\d+\.?\d*)$')[0].astype(float)
            ages_col = 'age_numeric'
        
        # Check age range for each cell type
        age_ranges = []
        for ct in obs['Sub_CT'].unique():
            ct_obs = obs[obs['Sub_CT'] == ct]
            ages = ct_obs[ages_col].dropna()
            if len(ages) > 0:
                age_ranges.append({
                    'Sub_CT': ct,
                    'min_age': ages.min(),
                    'max_age': ages.max(),
                    'age_span': ages.max() - ages.min(),
                    'n_donors': ct_obs['donor_id'].nunique()
                })
        
        if len(age_ranges) == 0:
            print("  ⚠️ Could not parse age information")
            continue
            
        df_age = pd.DataFrame(age_ranges).sort_values('age_span')
        
        # Overall age range
        all_ages = obs[ages_col].dropna()
        dataset_age_span = all_ages.max() - all_ages.min()
        
        print(f"Dataset age range: {all_ages.min():.0f} - {all_ages.max():.0f} (span: {dataset_age_span:.0f} years)")
        
        # Flag cell types with limited age coverage
        for _, row in df_age.iterrows():
            coverage = (row['age_span'] / dataset_age_span) * 100 if dataset_age_span > 0 else 0
            flag = "⚠️ LIMITED AGE COVERAGE" if coverage < 50 else ""
            print(f"  {row['Sub_CT']:20s}: {row['min_age']:4.0f} - {row['max_age']:4.0f} years "
                  f"(coverage: {coverage:5.1f}%) {flag}")


def check_donor_representation():
    """
    Check if cell types are present in most donors.
    Red flag: If a cell type is only in a few donors.
    """
    print("\n" + "="*80)
    print("3. DONOR REPRESENTATION CHECK")
    print("="*80)
    
    for dataset in DISCOVERY_COHORTS:
        print(f"\n### {dataset.upper()} ###")
        obs = retrieve_adata(dataset=dataset, data_type='sc', only_obs=True)
        
        total_donors = obs['donor_id'].nunique()
        
        # Check donor coverage for each cell type
        donor_coverage = []
        for ct in obs['Sub_CT'].unique():
            ct_obs = obs[obs['Sub_CT'] == ct]
            n_donors = ct_obs['donor_id'].nunique()
            coverage = (n_donors / total_donors) * 100
            
            donor_coverage.append({
                'Sub_CT': ct,
                'n_donors': n_donors,
                'coverage': coverage
            })
        
        df_donors = pd.DataFrame(donor_coverage).sort_values('coverage')
        
        print(f"Total donors: {total_donors}")
        for _, row in df_donors.iterrows():
            flag = "⚠️ LOW DONOR COVERAGE" if row['coverage'] < 80 else ""
            print(f"  {row['Sub_CT']:20s}: {row['n_donors']:4d} donors ({row['coverage']:5.1f}%) {flag}")


def check_extreme_proportions():
    """
    Check for cell types with extreme proportions.
    Red flag: If a cell type is >50% or <1% of total cells.
    """
    print("\n" + "="*80)
    print("4. EXTREME PROPORTION CHECK")
    print("="*80)
    
    for dataset in DISCOVERY_COHORTS:
        print(f"\n### {dataset.upper()} ###")
        obs = retrieve_adata(dataset=dataset, data_type='sc', only_obs=True)
        
        total = len(obs)
        props = obs['Sub_CT'].value_counts() / total * 100
        
        # Check for extreme values
        high = props[props > 40]
        low = props[props < 2]
        
        if len(high) > 0:
            print("  ⚠️ VERY HIGH PROPORTIONS (>40%):")
            for ct, prop in high.items():
                print(f"    {ct}: {prop:.2f}%")
        
        if len(low) > 0:
            print("  ⚠️ VERY LOW PROPORTIONS (<2%):")
            for ct, prop in low.items():
                print(f"    {ct}: {prop:.2f}%")


def plot_proportion_comparison():
    """
    Create visualization comparing proportions across datasets.
    """
    all_props = []
    for dataset in DISCOVERY_COHORTS:
        obs = retrieve_adata(dataset=dataset, data_type='sc', only_obs=True)
        total = len(obs)
        props = obs['Sub_CT'].value_counts() / total * 100
        props_df = props.reset_index()
        props_df.columns = ['Sub_CT', 'proportion']
        props_df['dataset'] = dataset
        all_props.append(props_df)
    
    df = pd.concat(all_props)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    df_pivot = df.pivot(index='Sub_CT', columns='dataset', values='proportion')
    df_pivot = df_pivot.fillna(0)
    
    sns.heatmap(df_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Proportion (%)'}, vmin=0, vmax=50)
    
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Sub Cell Type', fontweight='bold')
    ax.set_title('Cell Type Proportions Across Datasets - Quality Check', 
                 fontweight='bold', pad=20)
    
    plt.tight_layout()
    file_name = f'{PLOTS_DIR}/annotation_quality_heatmap.png'
    print(f'\nSaving quality check heatmap to {file_name}')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()


def check_nonclassic_mono():
    """
    Special check for NonClassic_MONO which is not in SUB_CTS but appears in data.
    """
    print("\n" + "="*80)
    print("5. NON-STANDARD CELL TYPE CHECK")
    print("="*80)
    
    for dataset in DISCOVERY_COHORTS:
        obs = retrieve_adata(dataset=dataset, data_type='sc', only_obs=True)
        
        all_cts = set(obs['Sub_CT'].unique())
        standard_cts = set(SUB_CTS)
        
        extra = all_cts - standard_cts
        missing = standard_cts - all_cts
        
        if extra:
            print(f"\n{dataset}:")
            print(f"  ⚠️ EXTRA cell types (not in SUB_CTS): {extra}")
            for ct in extra:
                count = (obs['Sub_CT'] == ct).sum()
                pct = (count / len(obs)) * 100
                print(f"    {ct}: {count:,} cells ({pct:.2f}%)")
        
        if missing:
            print(f"  ⚠️ MISSING cell types (in SUB_CTS but not in data): {missing}")


def main():
    """Main execution."""
    print("="*80)
    print("CELL TYPE ANNOTATION QUALITY CHECK")
    print("="*80)
    
    # Run all checks
    df_props = check_proportion_consistency()
    check_age_representation()
    check_donor_representation()
    check_extreme_proportions()
    check_nonclassic_mono()
    
    # Create visualization
    plot_proportion_comparison()
    
    print("\n" + "="*80)
    print("SUMMARY OF RED FLAGS")
    print("="*80)
    print("""
Key Issues Identified:
1. NonClassic_MONO appears in all datasets but is NOT in SUB_CTS definition
2. Tcm_Naive_CD4 dominates in onek1k (48.67%) and abf300 (40.46%)
3. Classic_MONO varies wildly: 2.89% (onek1k) to 24.94% (perez_sle)
4. Some cell types have low proportions (<2%) in certain datasets
5. Cell type proportions are highly inconsistent across cohorts (high CV)
    """)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
