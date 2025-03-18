library(Signac)
library(SummarizedExperiment)
library(Matrix)
library(Seurat)
library(GenomicRanges)
library(ggplot2)
library(patchwork)
library(AnnotationHub)


task_grn_inference_dir = 'task_grn_inference/'
par <- list(
  atac = paste0(task_grn_inference_dir, "resources/grn_benchmark/inference_data/op_atac.h5ad") 
)
# --------------------- read the data and create a Seurat object
adata <- anndata::read_h5ad(par$atac)
counts <- t(adata$X)  # Transpose to match R's column-major order
rownames(counts) <- rownames(adata$var)
colnames(counts) <- rownames(adata$obs)
colData <- as.data.frame(adata$obs)

chrom_assay <- CreateChromatinAssay(
  counts = counts,
  sep = c(":", "-"),
  min.cells = 10,
  min.features = 200
)

atac <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = colData
)

#  keep only standard chromosomes
granges(atac)
peaks.keep <- seqnames(granges(atac)) %in% standardChromosomes(granges(atac))
atac <- atac[as.vector(peaks.keep), ]

# ----------------- add gene annotations
ah <- AnnotationHub()
query(ah, "EnsDb.Hsapiens.v98")
ensdb_v98 <- ah[["AH75011"]]
# extract gene annotations from EnsDb
annotations <- GetGRangesFromEnsDb(ensdb = ensdb_v98)
# change to UCSC style since the data was mapped to hg38
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- "hg38"
# add the gene information to the object
Annotation(atac) <- annotations


# -------------------- add blacklist ratio
atac$blacklist_ratio <- FractionCountsInRegion(
  object = atac, 
  assay = 'peaks',
  regions = blacklist_hg38_unified
)
VlnPlot(
  object = atac,
  features = c('blacklist_ratio'),
  pt.size = 0.1,
  ncol = 5
)
atac <- subset(
  x = atac,
    blacklist_ratio < 0.01 
)

# - umap
if (FALSE){
  atac <- RunTFIDF(atac)  # Normalize ATAC-seq counts
  atac <- FindTopFeatures(atac, min.cutoff = 10)  # Select variable peaks
  atac <- RunSVD(atac)  # Perform SVD instead of PCA for scATAC-seq
  atac <- RunUMAP(atac, reduction = "lsi", dims = 1:30)  # Reduce dimensions
  DimPlot(atac, reduction = "umap", group.by = "cell_type")  # Visualize UMAP
}

# ----------------- gene actitivity
gene.activities <- GeneActivity(atac)

# peaks <- CallPeaks(
#   object = atac,
#   group.by = "cell_type"
# )

CoveragePlot(
  object = atac,
  region = "CD8A",
  extend.upstream = 1000,
  extend.downstream = 1000,
  ranges.title = "MACS2",
  annotation = FALSE,
  peaks = FALSE
)

gene_plot <- AnnotationPlot(
  object = atac,
  region = "CD8A"
)
gene_plot

rownames(atac) <- sub("-", ":", rownames(atac))
peak_plot <- PeakPlot(
  object = atac,
  region = "CD8A"
)
peak_plot