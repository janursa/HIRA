# Predict age with the published scImmuAging clock. Run inside singularity/scimmuaging.sif.
# Usage: Rscript src/clock_benchmark/predict.R <pseudocells.tsv.gz> <cell_type> <out.tsv>
suppressPackageStartupMessages({
  library(scImmuAging); library(glmnet); library(data.table); library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
input <- args[1]; cell_type <- args[2]; out <- args[3]

models <- readRDS(system.file("data", "all_model.RDS", package = "scImmuAging"))
features <- readRDS(system.file("data", "all_model_inputfeatures.RDS", package = "scImmuAging"))
stopifnot(cell_type %in% names(models))

# zcat: fread would need R.utils to open a .gz itself. Genes not in df are zero-filled.
df <- as.data.frame(fread(cmd = paste("zcat", shQuote(input))))  # donor_id, age, <ENSG...>
pred <- AgingClockCalculator(df, models[[cell_type]], features[[cell_type]])

# Age_Donor() is the package's own per-donor aggregation: round(mean) over the pseudocells.
Age_Donor(pred) %>%
  dplyr::select(donor_id, age, predicted_age = predicted) %>%
  distinct() %>%
  mutate(cell_type = cell_type) %>%
  fwrite(out, sep = "\t")
cat("wrote", out, "\n")
