
args <- commandArgs(trailingOnly = TRUE)
file_path <- args[1] 
save_path <- args[2]  

df <- read.csv(file_path)

# Perform meta-analysis
library(dplyr)
library(metap)


# meta_results <- df %>%
#   group_by(gene) %>%
#   summarise(meta_p = sumlog(pvalue)$p, ) %>%
#   mutate(meta_p_adj = p.adjust(meta_p, method = "BH"))

meta_results <- df %>%
  group_by(gene) %>%
  summarise(meta_p = max(pvalue)) %>%
  mutate(meta_p_adj = p.adjust(meta_p, method = "BH"))

# meta_results <- df %>%
#   group_by(gene) %>%
#   summarise(meta_p = min(pvalue * n())) %>%
#   mutate(meta_p_adj = p.adjust(meta_p, method = "BH"))

# Count significant genes

write.csv(meta_results, file = save_path, row.names = FALSE)