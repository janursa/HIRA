
library(Signac)
library(Seurat)
library(GenomicRanges)
library(ggplot2)
library(patchwork)
library(SeuratWrappers) # remotes::install_github("satijalab/seurat-wrappers", dependencies = TRUE, ask = FALSE)
library(cicero)


# Define mapping of fine-grained cell types to major cell types
cell_type_mapping <- c(
  "CD14+ Monocytes" = "MONO",
  "CD4 Naive" = "CD4T",
  "B cell progenitor" = "B",
  "CD4 Memory" = "CD4T",
  "NK dim" = "NK",
  "CD8 Naive" = "CD8T",
  "pre-B cell" = "B",
  "CD8 effector" = "CD8T",
  "Double negative T cell" = "CD4T",  # Assuming helper-like behavior, adjust if needed
  "NK bright" = "NK",
  "CD16+ Monocytes" = "MONO",
  "Dendritic cell" = "MONO",  # DCs are myeloid lineage, adjust if needed
  "pDC" = "MONO"  # Plasmacytoid dendritic cells, adjust if needed
)

# main_dir <- "/home/jnourisa/projs/ongoing/ciim/"
# print(sys.frame)
# script_dir <- dirname(sys.frame(1)$ofile)
par = list(
  pbmc_multiome = 'input/motif_analysis/pbmc_multiome.rds'
)

pbmc = readRDS(par$pbmc_multiome)
DefaultAssay(pbmc) <- 'peaks'

# Apply mapping
pbmc@meta.data$cell_type <- cell_type_mapping[pbmc@meta.data$predicted.id]

pbmc <- SortIdents(pbmc)

if (TRUE){ #TODO: for each cell type
  # convert to CellDataSet format and make the cicero object
  pbmc.cds <- as.cell_data_set(x = pbmc)

  pbmc.cicero <- make_cicero_cds(pbmc.cds, reduced_coordinates = reducedDims(pbmc.cds)$UMAP)
  # get the chromosome sizes from the Seurat object
  genome <- seqlengths(pbmc)

  # use chromosome 1 to save some time
  genome <- genome[1] #TODO: omit this step to run on the whole genome

  # convert chromosome sizes to a dataframe
  genome.df <- data.frame("chr" = names(genome), "length" = genome)

  # run cicero
  conns <- run_cicero(pbmc.cicero, genomic_coords = genome.df, sample_num = 100)
  print(head(conns))
}
aaa
#  -------------- Plotting genomic regions


# open_cd4naive <- rownames(da_peaks[da_peaks$avg_log2FC > 3, ])
# open_cd14mono <- rownames(da_peaks[da_peaks$avg_log2FC < -3, ])

# closest_genes_cd4naive <- ClosestFeature(pbmc, regions = open_cd4naive)
# closest_genes_cd14mono <- ClosestFeature(pbmc, regions = open_cd14mono)

# find DA peaks overlapping gene of interest
# regions_highlight <- subsetByOverlaps(StringToGRanges(open_cd4naive), LookupGeneCoords(pbmc, "CD4"))

# regions_highlight # GRanges object

# CoverageBrowser(object = pbmc,
#   region = "S100A4",
# #   region.highlight = regions_highlight,
#   extend.upstream = 1000,
#   extend.downstream = 1000)



# conn <- data.frame(
#   Peak1 = c("chr1-100003337-100003837", "chr1-100003337-100003837", "chr1-100003337-100003837",
#             "chr1-100003337-100003837", "chr1-100003337-100003837", "chr1-100003337-100003837"),
#   Peak2 = c("chr1-99791719-99792219", "chr1-99828699-99829199", "chr1-99835542-99836042",
#             "chr1-99836217-99836717", "chr1-99839576-99840076", "chr1-99840640-99841140"),
#   coaccess = c(0, 0, 0, 0, 0, 0)
# )

# ccans <- data.frame(
#   Peak = c("chr1-10009702-10010202", "chr1-100151188-100151688", "chr1-100164787-100165287",
#             "chr1-100165566-100166066", "chr1-100202505-100203005", "chr1-100215491-100215991"),
#  row.names = c("chr1-10009702-10010202", "chr1-100151188-100151688", "chr1-100164787-100165287",
#                 "chr1-100165566-100166066", "chr1-100202505-100203005", "chr1-100215491-100215991"),
#   CCAN = c(1, 2, 2, 2, 3, 3)
# )

# links <- ConnectionsToLinks(conns = conns, ccans=ccans)

# Links(pbmc) <- gr


cov_plot <- CoveragePlot(
  object = pbmc,
  region = "TCF7L2",
    # region = StringToGRanges(c('chr10-133-137')),
  assay = 'peaks',
group.by = "cell_type",  # Use cell type for grouping

#   region.highlight = regions_highlight,
#   extend.upstream = 10000,
#   extend.downstream = 10000,
#   links = TRUE
)
ggsave("CoveragePlo.png", plot = cov_plot, width = 8, height = 6, dpi = 300)




# regions_highlight <- subsetByOverlaps(StringToGRanges(open_cd14mono), LookupGeneCoords(pbmc, "LYZ"))

# CoveragePlot(
#   object = pbmc,
#   region = "LYZ",
#   region.highlight = regions_highlight,
#   extend.upstream = 1000,
#   extend.downstream = 5000
# )

# # not run
# # peak_ranges should be a set of genomic ranges spanning the set of peaks to be quantified per cell
# peak_matrix <- FeatureMatrix(
#   fragments = Fragments(pbmc),
#   features = peak_ranges
# )

# # not run
# bin_matrix <- GenomeBinMatrix(
#   fragments = Fragments(pbmc),
#   genome = seqlengths(pbmc),
#   binsize = 5000
# )

# # not run
# total_fragments <- CountFragments('10k_pbmc_ATACv2_nextgem_Chromium_Controller_fragments.tsv.gz')
# rownames(total_fragments) <- total_fragments$CB
# pbmc$fragments <- total_fragments[colnames(pbmc), "frequency_count"]

# pbmc <- FRiP(
#   object = pbmc,
#   assay = 'peaks',
#   total.fragments = 'fragments'
# )

# # not run
# pbmc$blacklist_fraction <- FractionCountsInRegion(
#   object = pbmc, 
#   assay = 'peaks',
#   regions = blacklist_hg38_unified
# )

# saveRDS(object = pbmc, file = "pbmc.rds")
