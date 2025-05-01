# =====================================================================
#  AgeingClock Application – LGBM_SHAPtuneNS & LGBMtune
# ---------------------------------------------------------------------
#  Author : 
#  Updated: 2025‑04‑28
# ---------------------------------------------------------------------
# =====================================================================

# ---------------------------
# 0. Library loader
# ---------------------------
required_pkgs <- c("Seurat", "dplyr", "purrr", "tibble", "readr", "data.table",
                   "Matrix", "rlang", "stringr", "lightgbm", "xgboost", "glmnet", "tidyr")
install_if_missing <- function(pkgs) {
  inst <- rownames(installed.packages())
  for (p in pkgs) if (!p %in% inst) install.packages(p, repos="https://cloud.r-project.org")
}
install_if_missing(required_pkgs)

suppressPackageStartupMessages(lapply(required_pkgs, library, character.only = TRUE))

# ---------------------------
# 1. Helper utilities
# ---------------------------
get_model_type <- function(model) {
  cls <- class(model)[1]
  if      (cls %in% c("lgb.Booster", "lightgbm.Booster")) return("lightgbm")
  else if (cls == "xgb.Booster")                             return("xgboost")
  else if (cls %in% c("cv.glmnet", "glmnet"))              return("glmnet")
  stop("[E] Unsupported model class: ", cls)
}

make_feature_matrix <- function(df_expr, feature_order) {
  miss <- setdiff(feature_order, colnames(df_expr))
  if (length(miss) > 0) df_expr[miss] <- 0
  as.matrix(df_expr[, feature_order, drop = FALSE])
}

predict_age <- function(model, mtx) {
  switch(get_model_type(model),
         lightgbm = predict(model, mtx, num_iteration = model$best_iter, predict_disable_shape_check = TRUE),
         xgboost  = xgboost::predict(model, mtx),
         glmnet   = {
           lambda <- ifelse(!is.null(model$lambda.min), model$lambda.min, min(model$lambda))
           as.vector(predict(model, newx = mtx, s = lambda))
         })
}

bootstrap_pseudocells <- function(df, size = 15, n = 100, replace = "dynamic") {
  if (replace == "dynamic") {
    replace <- nrow(df) <= size
  }
  pseudocells <- c()
  for (i in 1:n) {
    batch <- df[sample(1:nrow(df), size = size, replace = replace), ]
    pseudocells <- rbind(pseudocells, colMeans(batch))
  }
  colnames(pseudocells) <- colnames(df)
  return(as_tibble(pseudocells))
}

# ---------------------------
# 2. Metadata harmonisation
# ---------------------------
harmonise_metadata <- function(svz, mapping = NULL) {
  meta <- svz@meta.data

  # default mapping if none supplied
  if (is.null(mapping)) {
    mapping <- c(
      donor_id      = "donor_id",
      disease       = "disease",
      disease_state = "disease_state",
      age           = "age",
      sex           = "sex",
      race          = "self_reported_ethnicity",
      Major_CT      = "Major_CT"
    )
  }

  present <- intersect(names(mapping), colnames(meta))
  if (length(present) > 0) {
    meta <- dplyr::rename(meta, !!!rlang::set_names(mapping[present], present))
  }

  svz@meta.data <- meta
  svz
}

# ---------------------------
# 3. Pre‑processing
# ---------------------------
Convert_to_Dataframe <- function(svz) {
  DefaultAssay(svz) <- "RNA"
  meta <- svz@meta.data
  meta <- meta[, c("donor_id", "age", 'sex', 'disease', 'disease_state', 'race')]
  raw_counts <- t(as.matrix(svz[["RNA"]]$data))
  df <- as_tibble(cbind(meta, raw_counts))
  return(df)
}

Preprocess_data <- function(svz, ct, markers, size = 15, n_boot = 100, ct_col = "Major_CT", study_name = NULL, outdir = NULL) {
  if (!ct_col %in% colnames(svz@meta.data)) stop("[E] Cell‑type column ", ct_col, " missing")
  cells_use <- rownames(svz@meta.data)[svz@meta.data[[ct_col]] == ct]
  if (length(cells_use) == 0) stop("[E] No cells of type ", ct)

  # Check if bootstrapped data already exists
  if (!is.null(outdir) && !is.null(study_name)) {
    bootstrap_dir <- file.path(outdir, "bootstrapping")
    bootstrap_file <- file.path(bootstrap_dir, paste0(study_name, "_", ct, "_bootstrapped.rds"))
    
    if (file.exists(bootstrap_file)) {
      message("[✓] Loading existing bootstrapped data from ", bootstrap_file)
      df_sc <- readRDS(bootstrap_file)
      df_sc$data <- NULL
      df_proc <- df_sc %>% tidyr::unnest(pseudo)
      
      return(list(expr = df_proc %>% select(-c(donor_id, age, sex, disease, disease_state, race)), 
                 meta = df_proc[, c("donor_id", "age", "sex", "disease", "disease_state", "race")]))
    } else {
      data_sub <- subset(svz, cells = cells_use)
      
      # Convert Seurat object to dataframe and generate pseudocells
      df_sc <- Convert_to_Dataframe(data_sub) %>%
        group_by(donor_id, age, sex, disease, disease_state, race) %>%
        tidyr::nest()
      
      df_sc <- df_sc %>%
        mutate(pseudo = map(data, ~ bootstrap_pseudocells(.x, size, n_boot)))
      
      # Save bootstrapping results before removing data column
      if (!dir.exists(bootstrap_dir)) dir.create(bootstrap_dir, recursive = TRUE)
      saveRDS(df_sc, bootstrap_file)
      message("[✓] Saved bootstrapping results to ", bootstrap_file)
    }
  } else {
    data_sub <- subset(svz, cells = cells_use)
    
    # Convert Seurat object to dataframe and generate pseudocells
    df_sc <- Convert_to_Dataframe(data_sub) %>%
      group_by(donor_id, age, sex, disease, disease_state, race) %>%
      tidyr::nest()
    
    df_sc <- df_sc %>%
      mutate(pseudo = map(data, ~ bootstrap_pseudocells(.x, size, n_boot)))
  }
  
  df_sc$data <- NULL
  df_proc <- df_sc %>% tidyr::unnest(pseudo)
  
  list(expr = df_proc %>% select(-c(donor_id, age, sex, disease, disease_state, race)), 
       meta = df_proc[, c("donor_id", "age", "sex", "disease", "disease_state", "race")])
}

# ---------------------------
# 4. Predictors
# ---------------------------
AC_AgePredict_per_cell <- function(svz, ct, model, feature_order, ct_col = "Major_CT", study_name = NULL, outdir = NULL) {
  pp <- Preprocess_data(svz, ct, feature_order, ct_col = ct_col, study_name = study_name, outdir = outdir)
  mtx <- make_feature_matrix(pp$expr, feature_order)
  
  # Get model type and make predictions
  model_type <- get_model_type(model)
  if (model_type == "lightgbm") {
    preds <- predict(model, mtx, num_iteration = model$best_iter, predict_disable_shape_check = TRUE)
  } else if (model_type == "xgboost") {
    preds <- xgboost::predict(model, mtx)
  } else if (model_type == "glmnet") {
    lambda <- ifelse(!is.null(model$lambda.min), model$lambda.min, min(model$lambda))
    preds <- as.vector(predict(model, newx = mtx, s = lambda))
  }
  
  bind_cols(pp$meta, tibble(predicted_age = preds))
}

AC_AgePredict_per_donor <- function(svz, ct, model, feature_order, ct_col = "Major_CT", study_name = NULL, outdir = NULL) {
  cell_df <- AC_AgePredict_per_cell(svz, ct, model, feature_order, ct_col, study_name, outdir)
  cell_df %>%
    group_by(donor_id, age, sex, disease, disease_state, race) %>%
    summarise(pred_age = median(predicted_age), n_cells = n(), .groups = "drop")
}

# ---------------------------
# 5. Model / feature paths
# ---------------------------
resolve_files <- function(ct, model_choice, base_dir) {
  model_choice <- match.arg(model_choice, c("LGBMtune", "LGBM_SHAPtuneNS"))
  mdir <- file.path(base_dir, "models", "majorCT", "entire")
  fdir <- file.path(base_dir, "features", "majorCT", "entire")
  if (model_choice == "LGBMtune") {
    list(model = file.path(mdir, sprintf("%s_1_LGBMtune_model.RDS", ct)),
         features = file.path(fdir, "Correlation", sprintf("%s_1_gridsearch.RDS", ct)))
  } else {
    model_file <- file.path(mdir, sprintf("%s_5_LGBM_SHAPtuneNS_model.RDS", ct))
    pattern <- sprintf("%s_feature_consensus_[0-9]+cv.tsv", ct)
    feat_file <- list.files(file.path(fdir, "SHAPNS"), pattern = pattern, full.names = TRUE)[1]
    if (is.na(feat_file)) stop("[E] SHAPNS feature file not found for ", ct)
    list(model = model_file, features = feat_file)
  }
}

read_feature_vector <- function(path) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "rds")   vec <- if (is.list(v <- readRDS(path))) unlist(v[[length(v)]]) else v
  else if (ext %in% c("txt", "tsv")) {
    df <- readr::read_tsv(path, col_names = FALSE, show_col_types = FALSE)
    vec <- if (ncol(df) == 1) df[[1]] else { names(df)[1:2] <- c("gene", "count"); ifelse(df$count >= 6, df$gene, NA) }
  } else stop("[E] Unsupported feature file: ", path)
  unique(na.omit(vec))
}

get_model_features <- function(model) {
  type <- get_model_type(model)
  if (type == "lightgbm") {
    imp <- tryCatch(lightgbm::lgb.importance(model), error = function(e) NULL)
    if (!is.null(imp)) return(unique(imp$Feature))
    return(model$.__enclos_env__$private$feature_name)
  } else if (type == "xgboost") {
    return(model$feature_names)
  } else if (type == "glmnet") {
    return(rownames(model$beta))
  } else character()
}

load_model_file <- function(path) {
  if (grepl("LGBM", basename(path), ignore.case = TRUE)) {
    m <- tryCatch(lightgbm::lgb.load(path), error = function(e) NULL)
    if (!is.null(m)) return(m)
  }
  readRDS(path)
}

# ---------------------------
# 6. Age prediction calculation
# ---------------------------
stats2 <- function(predictions_df) {
  d <- as_tibble(predictions_df)
  
  d2 <- d %>% 
    group_by(donor_id, age, sex, disease, disease_state, race) %>%
    mutate(med = median(predicted_age)) %>% 
    distinct()
  
  d2$age <- as.numeric(d2$age)
  mae <- round(median(abs(d2$age - d2$med)), 3)
  meanae <- round(mean(abs(d2$age - d2$med)), 3)
  rho <- round(cor(d2$age, d2$med, method = "pearson"), 3)
  rmse <- sqrt(mean((d2$age - d2$med)^2))
  
  message("correlation: ", rho)
  message("RMSE: ", rmse)
  message("Median Absolute Error: ", mae)
  message("Mean Absolute Error: ", meanae)
  
  return(list(correlation = rho, RMSE = rmse, MAE = mae, MeanAE = meanae, data = d2))
}

# ---------------------------
# 7. Runner
# ---------------------------
run_AgeingClock <- function(seurat_rds, ct, model_choice = c("LGBMtune", "LGBM_SHAPtuneNS"),
                            outdir = "./AC_results", study_name = NULL,
                            base_dir = "/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/out_for_application",
                            donor_id_col = "donor_id", age_col = "age", sex_col = NULL,
                            disease_col = NULL, disease_state_col = NULL, race_col = NULL,
                            ct_col = "Major_CT") {

  model_choice <- match.arg(model_choice)
  if (!dir.exists(outdir)) dir.create(outdir, TRUE)

  # remap metadata columns
  map <- c(); map[donor_id_col] <- "donor_id"; map[age_col] <- "age"
  if (!is.null(sex_col)) map[sex_col] <- "sex"
  if (!is.null(disease_col)) map[disease_col] <- "disease"
  if (!is.null(disease_state_col)) map[disease_state_col] <- "disease_state"
  if (!is.null(race_col)) map[race_col] <- "race"
  if (ct_col != "Major_CT") map[ct_col] <- "Major_CT"

  svz <- if (is.character(seurat_rds)) readRDS(seurat_rds) else seurat_rds
  svz <- harmonise_metadata(svz, map)

  stopifnot(all(c("donor_id", "age", "Major_CT") %in% colnames(svz@meta.data)))
  if (!ct %in% svz@meta.data$Major_CT) stop("[E] Cell‑type not in data")

  paths <- resolve_files(ct, model_choice, base_dir)
  model <- load_model_file(paths$model)
  model_features <- get_model_features(model)
  features_file <- read_feature_vector(paths$features)
  feature_order <- unique(c(model_features, features_file))

  cell_df <- AC_AgePredict_per_cell(svz, ct, model, feature_order, ct_col, study_name, outdir)
  donor_df <- AC_AgePredict_per_donor(svz, ct, model, feature_order, ct_col, study_name, outdir)

  # Calculate statistics
  stats <- stats2(cell_df)
  
  # Format results for output
  if (!is.null(study_name)) { 
    cell_df$study <- study_name
    donor_df$study <- study_name 
  }
  
  prefix <- paste(na.omit(c(study_name, ct, model_choice)), collapse = "_")
  readr::write_tsv(cell_df, file.path(outdir, paste0(prefix, "_cell.tsv")))
  readr::write_tsv(donor_df, file.path(outdir, paste0(prefix, "_donor.tsv")))

  # Print metrics to console
  ds <- donor_df
  message(sprintf("[i] Donor Pearson=%.3f | RMSE=%.3f | MAE=%.3f",
              cor(ds$age, ds$pred_age, method="pearson"),
              sqrt(mean((ds$age - ds$pred_age)^2)),
              median(abs(ds$age - ds$pred_age))))
  message("[✓] Results saved to ", outdir)
  
  invisible(list(cell = cell_df, donor = donor_df, stats = stats))
}

# =====================================================================
#  End of script
# =====================================================================



## Example usage
# dat <- readRDS('/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/outputs/application_data_processed/MS_processed.rds')
dat <- readRDS('/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/outputs/application_data_processed/MS_processed.h5ad')


data_converter ->  function(data_h5ad){
  print('Input data should be normalized. .X')
  X = data_h5ad.X  

  # - pseudocells
  return data_rds
}


# TODO: wrapper
wrapper_run_AgeingClock_donors(dat, 
                cell_types=['MONO','B'],  
                model_choice = "LGBM_SHAPtuneNS", 
                study_name = "MS",
                outdir = "/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/out_for_application/test/AC_results", 
                base_dir = "/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/out_for_application", 
                donor_id_col = "donor_id_age",
                ct_col = "Major_CT" 
                )

#TODO: minimal code 
run_AgeingClock(dat, 
                "MONO",  #required 
                model_choice = "LGBM_SHAPtuneNS", 
                study_name = "MS",
                outdir = "/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/out_for_application/test/AC_results", 
                base_dir = "/vol/projects/aehsani/ImmuneAgeing/Immuneageing_vF/out_for_application", 
                ct_col = "Major_CT" 
                )
