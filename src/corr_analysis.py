import argparse

from helper import *


def main():
    parser = argparse.ArgumentParser(description="Run co-expression calculations and significance tests with customizable parameters.")
    
    # Define arguments with default values
    parser.add_argument('--normalize', type=str, default='sla', help="Normalization method")
    parser.add_argument('--corr_method', type=str, default='pearson', help="Correlation method")
    parser.add_argument('--denoise', action='store_true', help="Enable denoising if specified")    
    # Parse arguments
    args = parser.parse_args()

    # Generate the filenames based on parsed arguments
    adata_dir = 'input/adata_bootstrapped.h5ad'
    coexp_adata_file = f'output/coexp_adata_{args.normalize}_{args.corr_method}_{args.denoise}.h5ad'
    save_file = f'output/links_pvalues_vs_34_{args.normalize}_{args.corr_method}_{args.denoise}.csv'

    # Call functions with the generated arguments
    calculate_coexp_all(
        adata_dir=adata_dir,
        normalize=args.normalize,
        corr_method=args.corr_method,
        write_file=coexp_adata_file,
        denoise=args.denoise,
        targeted=True
    )

    sig_test_all(
        coexp_adata_file=coexp_adata_file,
        ctr_group='34-',
        col_contrast='age_group',
        col_link='link',
        save_file=save_file
    )

if __name__ == "__main__":
    main()