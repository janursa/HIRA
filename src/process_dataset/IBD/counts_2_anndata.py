import scanpy as sc
import pandas as pd
from scipy import io

data_dir = '/vol/projects/HIARA/processed/multiome/IBD/data/'
to_save = '/vol/projects/HIARA/processed/multiome/IBD/'

for data_type in ['rna', 'atac']:

    # Load counts (Matrix Market format)
    X = io.mmread(f"{data_dir}/{data_type}_counts.mtx").T.tocsr()  # transpose to cells × peaks

    # Load peaks (features / rownames of ATAC assay)
    peaks = pd.read_csv(f"{data_dir}/{data_type}_features.csv", header=None)[0].values
    var = pd.DataFrame(index=peaks)

    # Load cell barcodes
    cells = pd.read_csv(f"{data_dir}/{data_type}_cells.csv", header=None)[0].values
    obs = pd.DataFrame(index=cells)

    # Load metadata (make sure it has a 'cell_id' column)
    meta = pd.read_csv(f"{data_dir}/{data_type}_metadata.csv")
    meta = meta.set_index("cell_id")
    obs = obs.join(meta)

    # Build AnnData
    adata = sc.AnnData(X=X, obs=obs, var=var)

    # Save to h5ad
    adata.write(f"{to_save}/{data_type}.h5ad")
