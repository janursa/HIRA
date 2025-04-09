library(ggplot2)
library(pheatmap)
library(dplyr)
library(tidyr)
library(tibble)

script_dir <- dirname(sys.frame(1)$ofile)
significant_data <- read.csv(file.path(script_dir, '../../output/tf_activation/stats_bulk_consistent.csv'), 
               header = TRUE, stringsAsFactors = FALSE)


# Step 2: Create a matrix for the heatmap
# Pivot the data so that each row is a TF, columns are cell types, and the values are based on trend
heatmap_matrix <- significant_data %>%
  select(tf, cell_type, trend) %>%
  spread(key = cell_type, value = trend) 

# Replace all remaining NA values with 0
heatmap_matrix <- heatmap_matrix %>%
  column_to_rownames(var = "tf")

# Check for missing values and replace with 0 if needed
heatmap_matrix[is.na(heatmap_matrix)] <- 0

# Ensure that all data is numeric
heatmap_matrix <- apply(heatmap_matrix, 2, as.numeric)

# Create the heatmap
pheatmap(heatmap_matrix, scale = "none", 
         clustering_distance_rows = "euclidean", 
         clustering_method = "complete", 
         color = colorRampPalette(c("blue", "white", "red"))(100),
         show_rownames = TRUE, # Display row names (TFs)
         show_colnames = TRUE  # Display column names (Cell types)
)
