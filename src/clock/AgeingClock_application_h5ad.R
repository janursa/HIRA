# ================================================================
# 0.  Packages  ---------------------------------------------------
# ================================================================
required <- c("Seurat", "SeuratObject",
              "SingleCellExperiment", "SummarizedExperiment",
              "reticulate", "Matrix",
              "dplyr", "purrr", "tibble", "tidyr", "readr", "data.table",
              "rlang", "stringr", "lightgbm", "xgboost", "glmnet")

# inst <- rownames(installed.packages())
# for (p in setdiff(required, inst))
#   install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(
  lapply(required, library, character.only = TRUE)
)
library(reticulate)
reticulate::use_python("~/miniconda3/envs/py10/bin/python", required=T)

np <- tryCatch(reticulate::import("numpy",   convert = FALSE), silent = TRUE)
an <- tryCatch(reticulate::import("anndata", convert = FALSE), silent = TRUE)

is_seurat  <- function(x) inherits(x, "Seurat")
is_sce     <- function(x) inherits(x, "SingleCellExperiment")
is_anndata <- function(x) inherits(x, "python.builtin.object") &&
                           !is.null(tryCatch(x$X, error = function(e) NULL))

# ------------------------------------------------
# 0‑b. Loader
# ------------------------------------------------
load_svz <- function(o) {
  if (is.character(o) && grepl("\\.h5ad$", o, ignore.case = TRUE))
    return(reticulate::import("scanpy", convert = FALSE)$read_h5ad(o))
  if (is.character(o) && grepl("\\.rds$",  o, ignore.case = TRUE))
    return(readRDS(o))
  o
}

# ================================================================
# 1.  Generic helpers
# ================================================================
get_model_type <- function(m)
  c(lightgbm="lightgbm", lgb.Booster="lightgbm",
    `lightgbm.Booster`="lightgbm",
    `xgb.Booster`="xgboost",
    cv.glmnet="glmnet", glmnet="glmnet")[class(m)[1]]

make_feature_matrix <- function(df, features) {
  features <- unique(na.omit(features))
  miss <- setdiff(features, colnames(df))
  if (length(miss)) df[, miss] <- 0
  as.matrix(df[, features, drop = FALSE])
}

predict_age <- function(model, mtx) switch(
  get_model_type(model),
  lightgbm = predict(model, mtx,
                     num_iteration = ifelse(is.null(model$best_iter), -1, model$best_iter)),
  xgboost  = xgboost::predict(model, mtx),
  glmnet   = {
    lam <- ifelse(!is.null(model$lambda.min), model$lambda.min, min(model$lambda))
    as.double(predict(model, newx = mtx, s = lam))
})

bootstrap_pseudocells <- function(df, size = 15, n = 100) {
  repl <- nrow(df) <= size
  as_tibble(do.call(rbind,
    lapply(seq_len(n), \(.) colMeans(df[sample(nrow(df), size, repl), ]))))
}

# ================================================================
# 2.  Metadata helpers (unchanged)
# ================================================================
get_meta <- function(o)
  if (is_seurat(o))            o@meta.data else
  if (is_sce(o))               as.data.frame(SummarizedExperiment::colData(o)) else
  if (is_anndata(o))           reticulate::py_to_r(o$obs) else
  stop("Unsupported object")

set_meta <- function(o, df) {
  if (is_seurat(o))      o@meta.data <- df else
  if (is_sce(o))         SummarizedExperiment::colData(o) <- S4Vectors::DataFrame(df) else
  if (is_anndata(o))     o$obs <- reticulate::r_to_py(df, convert = TRUE)
  o
}

harmonise_metadata <- function(o, map) {
  meta <- get_meta(o)
  for (canon in names(map)) {
    old <- map[[canon]]
    if (!is.na(old) && old %in% colnames(meta) && canon != old)
      meta <- dplyr::rename(meta, !!canon := !!rlang::sym(old))
  }
  req <- c("donor_id","age","sex","condition","dataset","cell_type")
  for (c in req[!req %in% colnames(meta)]) meta[[c]] <- NA
  set_meta(o, meta)
}

# ================================================================
# 3.  Expression → tibble (robust AnnData code)
# ================================================================
uniqueify <- function(v) {
  dup <- duplicated(v)
  if (any(dup)) v[dup] <- paste0(v[dup],"__dup",ave(seq_along(v),v,FUN=seq)[dup])
  v
}
py_to_dense <- function(X) {
  if (inherits(X, "scipy.sparse._csr.csr_matrix") ||
      inherits(X, "scipy.sparse._csc.csc_matrix"))
    reticulate::py_to_r(X$toarray())
  else reticulate::py_to_r(X)
}

Convert_to_Dataframe <- function(obj) {

  if (is_seurat(obj)) {
    DefaultAssay(obj) <- "RNA"
    counts <- t(as.matrix(obj[["RNA"]]$data))
    meta   <- obj@meta.data

  } else if (is_sce(obj)) {
    counts <- t(as.matrix(SummarizedExperiment::assay(obj, 1)))
    meta   <- as.data.frame(SummarizedExperiment::colData(obj))

  } else if (is_anndata(obj)) {

    genes <- if (!is.null(obj$var_names$to_list))
               reticulate::py_to_r(obj$var_names$to_list())
             else if (!is.null(obj$var_names$to_numpy))
               reticulate::py_to_r(obj$var_names$to_numpy())
             else
               reticulate::py_to_r(obj$var_names)

    genes <- uniqueify(as.character(genes))
    n_cells <- nrow(reticulate::py_to_r(obj$obs))
    n_genes <- length(genes)

    counts <- as.matrix(py_to_dense(obj$X))

    if (all(dim(counts) == c(n_cells, n_genes))) {
      # ok
    } else if (all(dim(counts) == c(n_genes, n_cells))) {
      counts <- t(counts)
      message("[i] Transposed counts so that rows = cells, cols = genes")
    } else {
      stop("[E] adata$X dims ", paste(dim(counts), collapse = "×"),
           " do not match cells=", n_cells, " genes=", n_genes)
    }

    colnames(counts) <- genes
    meta <- reticulate::py_to_r(obj$obs)

  } else stop("Unsupported object in Convert_to_Dataframe()")

  as_tibble(cbind(meta, counts))
}

# ================================================================
# 4.  Pre‑processing  (Seurat bootstraps; SCE/AnnData do not)
# ================================================================
Preprocess_data <- function(svz, ct, size = 15, n_boot = 100) {
  meta <- get_meta(svz)
  idx  <- which(meta$cell_type == ct)
  if (!length(idx)) stop("No cells of type ", ct)
  meta_cols <- c("donor_id","age","sex","condition","dataset","cell_type")

  if (is_seurat(svz)) {
    Convert_to_Dataframe(svz)[idx, ] %>%
      group_by(across(all_of(meta_cols))) %>%
      tidyr::nest() %>%
      mutate(pseudo = purrr::map(data, ~ bootstrap_pseudocells(.x,size,n_boot))) %>%
      select(-data) %>%
      unnest(pseudo) -> df
  } else {
    df <- Convert_to_Dataframe(svz)[idx, ]
  }
  list(expr = df %>% select(-all_of(meta_cols)),
       meta = df %>% select( all_of(meta_cols)))
}

# ================================================================
# 5.  Model utilities
# ================================================================
resolve_files <- function(ct, choice, base){
  mdir<-file.path(base,"models","majorCT","entire")
  fdir<-file.path(base,"features","majorCT","entire")
  if(choice=="LGBMtune")
    list(model=file.path(mdir,sprintf("%s_1_LGBMtune_model.RDS",ct)),
         features=file.path(fdir,"Correlation",sprintf("%s_1_gridsearch.RDS",ct)))
  else{
    mf<-file.path(mdir,sprintf("%s_5_LGBM_SHAPtuneNS_model.RDS",ct))
    pat<-sprintf("%s_feature_consensus_[0-9]+cv.tsv",ct)
    ff<-list.files(file.path(fdir,"SHAPNS"),
                   pattern=pat, full.names=TRUE)[1]
    if(is.na(ff))stop("Feature file missing")
    list(model=mf, features=ff)
  }
}

read_feature_vector <- function(p){
  ext<-tolower(tools::file_ext(p))
  v<-if(ext=="rds") readRDS(p) else
      readr::read_tsv(p, col_names=FALSE, show_col_types=FALSE)[[1]]
  unique(na.omit(if(is.list(v)) unlist(v) else v))
}

get_model_features <- function(model) {
  typ <- get_model_type(model)
  if (typ == "lightgbm") {
    fn <- tryCatch(model$.__enclos_env__$private$feature_name,
                   error = function(e) character())
    if (length(fn) == 0 && !is.null(model$feature_name))
      fn <- model$feature_name
    if (length(fn) == 0) {
      imp <- tryCatch(lightgbm::lgb.importance(model), error = function(e) NULL)
      if (!is.null(imp) && "Feature" %in% names(imp))
        fn <- imp$Feature
    }
    return(unique(na.omit(fn)))
  }
  if (typ == "xgboost") return(unique(na.omit(model$feature_names)))
  if (typ == "glmnet")  return(unique(na.omit(rownames(model$beta))))
  character()
}

load_model_file <- function(p){
  if (grepl("LGBM",basename(p),ignore.case=TRUE)) {
    m<-tryCatch(lightgbm::lgb.load(p), error = \(e) NULL)
    if (!is.null(m)) return(m)
  }
  readRDS(p)
}

# ================================================================
# 6.  Predictors
# ================================================================
AC_cell  <- function(svz,ct,m,feats){
  pp <- Preprocess_data(svz,ct)
  pp$meta %>% mutate(predicted_age =
           predict_age(m, make_feature_matrix(pp$expr,feats)))
}

AC_donor <- function(svz,ct,m,feats){
  AC_cell(svz,ct,m,feats) %>%
    group_by(donor_id,age,sex,condition) %>%
    summarise(pred_age=median(predicted_age),
              n_cells=n(), .groups="drop")
}

# ================================================================
# 7.  Runner
# ================================================================
run_AgeingClock_LGBMSHAP <- function(
  svz, ct,
  model_choice = "LGBM_SHAPtuneNS",
  outdir = "./AC_results",
  study_name = NULL,
  base_dir,
  donor_id_col="donor_id", age_col="age",
  sex_col=NULL, condition_col=NULL, ct_col="cell_type") {

  model_choice <- match.arg(model_choice)
  svz <- load_svz(svz)

  map <- c(donor_id=donor_id_col, age=age_col)
  if (!is.null(sex_col))       map["sex"]       <- sex_col
  if (!is.null(condition_col)) map["condition"] <- condition_col
  if (ct_col!="cell_type")     map["cell_type"] <- ct_col
  svz <- harmonise_metadata(svz, map)

  files       <- resolve_files(ct, model_choice, base_dir)
  model       <- load_model_file(files$model)
  file_feats  <- read_feature_vector(files$features)
  model_feats <- get_model_features(model)

  # ---- choose feature vector (flexible across model types) -----
  num_feature <- tryCatch(model$.__enclos_env__$private$num_feature,
                          error = function(e) NA)

  num_feature <- if (length(num_feature)) num_feature[1] else NA  # ← ADD THIS

  if (!is.na(num_feature)) {
    if (length(model_feats) == num_feature) {
      feats <- model_feats
    } else if (length(file_feats) == num_feature) {
      message("[i] Feature file matches model’s num_feature → using it")
      feats <- file_feats
    } else {
      feats <- unique(c(model_feats, file_feats))
      if (length(feats) > num_feature)
        feats <- feats[1:num_feature]
      if (length(feats) < num_feature) {
        pad <- setdiff(c(model_feats, file_feats), feats)
        feats <- c(feats, pad[seq_len(num_feature - length(feats))])
      }
      message("[i] Built hybrid feature vector of length ", length(feats))
    }
  } else {
    # fallback when num_feature not stored
    feats <- if (length(model_feats)) model_feats else file_feats
  }
  # --------------------------------------------------------------

  extras <- setdiff(file_feats, feats)
  if (length(extras))
    message("[i] Ignoring ", length(extras), " extra features")

  cell_df  <- AC_cell (svz, ct, model, feats)
  donor_df <- AC_donor(svz, ct, model, feats)

  if (!is.null(study_name)) {
    cell_df$study  <- study_name
    donor_df$study <- study_name
  }
  if (!dir.exists(outdir)) dir.create(outdir, TRUE)
  prefix <- paste(na.omit(c(study_name, ct, model_choice)), collapse="_")
  readr::write_tsv(cell_df,  file.path(outdir, paste0(prefix,"_cell.tsv")))
  readr::write_tsv(donor_df, file.path(outdir, paste0(prefix,"_donor.tsv")))

  cors <- cor(donor_df$age, donor_df$pred_age)
  rmse <- sqrt(mean((donor_df$age - donor_df$pred_age)^2))
  mae  <- median(abs(donor_df$age - donor_df$pred_age))
  message(sprintf("[✓] %s | R=%.3f  RMSE=%.3f  MAE=%.3f", ct, cors, rmse, mae))
  invisible(list(cell=cell_df, donor=donor_df))
}

run_AgeingClock_LGBM <- function(
  svz, ct,
  model_choice = "LGBMtune",
  outdir = "./AC_results",
  study_name = NULL,
  base_dir,
  donor_id_col="donor_id", age_col="age",
  sex_col=NULL, condition_col=NULL, ct_col="cell_type") {

  model_choice <- match.arg(model_choice)
  svz <- load_svz(svz)

  map <- c(donor_id=donor_id_col, age=age_col)
  if (!is.null(sex_col))       map["sex"]       <- sex_col
  if (!is.null(condition_col)) map["condition"] <- condition_col
  if (ct_col!="cell_type")     map["cell_type"] <- ct_col
  svz <- harmonise_metadata(svz, map)

  files       <- resolve_files(ct, model_choice, base_dir)
  model       <- load_model_file(files$model)
  file_feats  <- read_feature_vector(files$features)
  model_feats <- get_model_features(model)

  # ----------------------------------------------------------------
  # choose feature vector (robust across model flavours)  ★ FINAL ★
  # ----------------------------------------------------------------
  num_feature <- tryCatch(model$.__enclos_env__$private$num_feature,
                          error = function(e) NA)
  num_feature <- if (length(num_feature)) num_feature[1] else NA

  if (!is.na(num_feature)) {

    if (length(model_feats) == num_feature) {
      feats <- model_feats                                # SHAP model (44)

    } else if (length(file_feats) == num_feature ||
              (length(file_feats) >= num_feature &&
                length(model_feats) <  num_feature)) {
      message("[i] Using feature file because it supplies the ",
              num_feature, " genes expected by the model")
      feats <- file_feats[1:num_feature]                  # tuned model (5 597)

    } else {                                              # hybrid fallback
      feats <- unique(c(model_feats, file_feats))
      feats <- head(feats, num_feature)
      message("[i] Built hybrid feature vector of length ", length(feats))
    }

  } else {  # ---------- num_feature missing  -----------------------
    if (length(file_feats) > length(model_feats)) {
      message("[i] Model lacks num_feature; using longer feature file list")
      feats <- file_feats
    } else {
      feats <- model_feats
    }
  }
  # ----------------------------------------------------------------

  extras <- setdiff(file_feats, feats)
  if (length(extras))
    message("[i] Ignoring ", length(extras), " extra features")

  cell_df  <- AC_cell (svz, ct, model, feats)
  donor_df <- AC_donor(svz, ct, model, feats)

  if (!is.null(study_name)) {
    cell_df$study  <- study_name
    donor_df$study <- study_name
  }
  if (!dir.exists(outdir)) dir.create(outdir, TRUE)
  prefix <- paste(na.omit(c(study_name, ct, model_choice)), collapse="_")
  readr::write_tsv(cell_df,  file.path(outdir, paste0(prefix,"_cell.tsv")))
  readr::write_tsv(donor_df, file.path(outdir, paste0(prefix,"_donor.tsv")))

  cors <- cor(donor_df$age, donor_df$pred_age)
  rmse <- sqrt(mean((donor_df$age - donor_df$pred_age)^2))
  mae  <- median(abs(donor_df$age - donor_df$pred_age))
  message(sprintf("[✓] %s | R=%.3f  RMSE=%.3f  MAE=%.3f", ct, cors, rmse, mae))
  invisible(list(cell=cell_df, donor=donor_df))
}


# ================================================================
# 8.  Example usage (comment out when sourcing) -------------------
# ================================================================
# Load the h5ad file
args <- commandArgs(trailingOnly = TRUE)
data_file <- args[1]  # e.g., "data12" or "CXCL9"
output_file <- args[2]
condition_col <- args[3]

# data_file <- "/vol/projects/jnourisa/datasets/data13_metacell.h5ad"
# output_file <- '../output/data13.csv'
# condition_col <- 'disease'


cat("Running AgeingClock on:", data_file, "\n")

adata <- reticulate::import("scanpy", convert = FALSE)$read_h5ad(data_file)

adata$obs
table(reticulate::py_to_r(adata$obs$cell_type$to_numpy()))



# - actual run
# cell_types <- unique(py_to_r(adata$obs$cell_type$to_numpy()))
cell_types <- ('CD8T')
cell_results_store <- list()
for (cell_type in cell_types) {
  message("Running AgeingClock for cell type: ", cell_type)
  res <- run_AgeingClock_LGBM(
    adata, cell_type,
    study_name    = "noname",
    outdir        = "../output/clock",
    base_dir      = "/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/out_for_application",
    donor_id_col  = "donor_id",
    age_col       = "age",
    sex_col       = "sex",
    condition_col = condition_col,
    ct_col        = "cell_type"
  )

  cell_results <- res$cell
  cell_results$cell_type <- cell_type
  cell_results_store[[cell_type]] <- cell_results
}
all_cell_results <- bind_rows(cell_results_store)

# Save to file
# output_file <- "../output/clock/all_cell_results.tsv"
write_tsv(all_cell_results, output_file)
# ================================================================