library(tidyverse)
library(Seurat)
library(SingleCellExperiment)
library(ggplot2)
library(ggpubr)
library(viridis)
library(tradeSeq)
library(glue)
library(slingshot)

library(tidyverse)
library(Seurat)
library(SingleCellExperiment)
library(ggplot2)
library(ggpubr)
library(viridis)
library(Rmisc)

library(tradeSeq)
library(RColorBrewer)

library(glue)
library(slingshot)
library(pheatmap)

ingeo_colours <- c("#0077BB","#33BBEE","#009988","#EE7733","#CC3311","#EE3377","#BBBBBB", "#5D54A4" )


setwd("/Volumes/lab-santoss/home/users/ingeo/Analysis/Single_cell_RNA_seq_projects/2020_fate_switching_OI/2022_finalised_analysis/Seurat_Slingshot_TradeSeq/all_samples")


##CSVs
assoRes <- read_csv("assoRes.csv")
#startRes <- read_csv("startRes.csv")
#endRes <- read_csv("endRes.csv")



##FDR functions
padjust_comparisons_assoRes <- function(df){
  df <- df %>% mutate(padj_1 = p.adjust(pvalue_1, "fdr"), .after = pvalue_1) %>%
    mutate(padj_2 = p.adjust(pvalue_2, "fdr"), .after = pvalue_2) %>%
    mutate(padj_3 = p.adjust(pvalue_3, "fdr"), .after = pvalue_3) %>%
    mutate(padj_4 = p.adjust(pvalue_4, "fdr"), .after = pvalue_4) %>%
    mutate(padj_5 = p.adjust(pvalue_5, "fdr"), .after = pvalue_5)
  
  return(df)
  
  
  
  
}

assoRes <- padjust_comparisons_assoRes(assoRes)

head(assoRes)

##AssoRes
assoRes_endo <- assoRes %>% filter(padj_1 < 0.05) %>% arrange(desc(waldStat_1))
assoRes_meso <- assoRes %>% filter(padj_2 < 0.05) %>% arrange(desc(waldStat_2))

assoRes_endo_s <- assoRes_endo %>% filter(!(`...1` %in% assoRes_meso$...1))


assoRes_meso_s <- assoRes_meso %>% filter(!(`...1` %in% assoRes_endo$...1))


## Heatmap

sce <- readRDS(file = "All_lineages_fitGAM.rds")

### based on mean smoother

Data <- rbind(assoRes_meso_s, assoRes_endo_s)

unique(Data$`...1`)



genes <- unique(Data$`...1`)


yhatSmooth_lin_1vs2 <- predictSmooth(sce, gene = genes, nPoints = 100, tidy = FALSE) %>%
  as.data.frame() %>% 
  dplyr::select(starts_with(c("lineage1_", "lineage2_")))
yhatSmoothScaled_lin_1vs2 <- t(apply(yhatSmooth_lin_1vs2, 1, scales::rescale))

breaksList <- seq(-1, 1, by = 0.2)


# Assuming yhatSmoothScaled_lin_1vs2 is your data matrix and you have a vector of genes to label
specific_genes <- c("HAND1", "SOX2", "GATA6")  # Replace with your specific genes

# Create a data frame for row annotations
row_annotation <- data.frame(Gene = rep("", nrow(yhatSmoothScaled_lin_1vs2)))  # Empty labels
rownames(row_annotation) <- rownames(yhatSmoothScaled_lin_1vs2)

# Mark the specific genes
row_annotation$Gene[rownames(row_annotation) %in% specific_genes] <- rownames(row_annotation)[rownames(row_annotation) %in% specific_genes]

# Generate the heatmap
pheatmap(yhatSmoothScaled_lin_1vs2,
         cluster_cols = FALSE,
         scale = "row",
         clustering_method = "ward.D",
         border_color = NA, 
         show_rownames = TRUE,
         show_colnames = FALSE,
         breaks = breaksList,
         color = colorRampPalette(rev(brewer.pal(n = 7, name = "RdBu")))(length(breaksList)),  
         cutree_rows = 8,
         annotation_row = row_annotation,
         annotation_names_row = TRUE)
