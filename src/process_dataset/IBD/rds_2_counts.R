library(Seurat)
library(Matrix)

# Load Seurat object
seurat_obj <- readRDS("/vol/projects/CIIM/processed/multiome/IBD/Seurat.rds")
SAVE_DIR <- "/vol/projects/CIIM/processed/multiome/IBD/data"

# Function to save one assay
save_assay <- function(seurat_obj, assay_name, SAVE_DIR) {
  assay <- seurat_obj[[assay_name]]
  
  # Write matrix
  Matrix::writeMM(assay@counts, file = file.path(SAVE_DIR, paste0(assay_name, "_counts.mtx")))
  
  # Write features (rows)
  write.table(
    rownames(assay@counts),
    file = file.path(SAVE_DIR, paste0(assay_name, "_features.csv")),
    quote = FALSE, row.names = FALSE, col.names = FALSE
  )
  
  # Write cells (columns)
  write.table(
    colnames(assay@counts),
    file = file.path(SAVE_DIR, paste0(assay_name, "_cells.csv")),
    quote = FALSE, row.names = FALSE, col.names = FALSE
  )
  
  # Metadata (add cell_id as first column)
  meta <- seurat_obj@meta.data
  meta$cell_id <- rownames(meta)
  write.csv(meta, file.path(SAVE_DIR, paste0(assay_name, "_metadata.csv")), row.names = FALSE)
  
  message(paste("Saved", assay_name, "to", SAVE_DIR))
}

# Run for both RNA and ATAC
save_assay(seurat_obj, "RNA", SAVE_DIR)
save_assay(seurat_obj, "ATAC", SAVE_DIR)