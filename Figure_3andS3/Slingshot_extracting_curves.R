library(tidyverse)
library(Seurat)
library(SingleCellExperiment)
library(ggplot2)
library(ggpubr)
library(viridis)
library(tradeSeq)
library(glue)
library(slingshot)
ingeo_colours <- c("#0077BB","#33BBEE","#009988","#EE7733","#CC3311","#EE3377","#BBBBBB", "#A020F0")




WT_int_seurat <- readRDS(file = "FS_seurat_converted_raw_count.rds")

WT_int_seurat

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$louvain

sce_WT <- as.SingleCellExperiment(WT_int_seurat, assay = "RNA")

table(sce_WT$louvain)

FeaturePlot(object = WT_int_seurat, features ="SOX17")

sce_WT_cl0 <- slingshot(sce_WT,
                        clusterLabels = "louvain",
                        start.clus = "0",
                        end.clus = c("11","4","6","2"),
                        approx_points = 200,
                        extend = "n",
                        reducedDim = "UMAP", stretch=0)




slingLineages(sce_WT_cl0)


sce_WT_cl0$slingPseudotime_1

# Add the Pseudotime information from Slingshot to the Seurat object
WT_int_seurat$Endo_pt <- sce_WT_cl0$slingPseudotime_1
WT_int_seurat$Meso_pt <- sce_WT_cl0$slingPseudotime_2
WT_int_seurat$PGCLC_pt <- sce_WT_cl0$slingPseudotime_3
WT_int_seurat$Amnion_pt <- sce_WT_cl0$slingPseudotime_4
WT_int_seurat$Pluri_pt <- sce_WT_cl0$slingPseudotime_5

WT_int_seurat$cl0_slingPseudotime_1 <- sce_WT_cl0$slingPseudotime_1
WT_int_seurat$cl0_slingPseudotime_2 <- sce_WT_cl0$slingPseudotime_2
WT_int_seurat$cl0_slingPseudotime_3 <- sce_WT_cl0$slingPseudotime_3
WT_int_seurat$cl0_slingPseudotime_4 <- sce_WT_cl0$slingPseudotime_4
WT_int_seurat$cl0_slingPseudotime_5 <- sce_WT_cl0$slingPseudotime_5

l1 <- as.data.frame(slingCurves(sce_WT_cl0)[[1]]$s)
l2 <- as.data.frame(slingCurves(sce_WT_cl0)[[2]]$s)
l3 <- as.data.frame(slingCurves(sce_WT_cl0)[[3]]$s)
l4 <- as.data.frame(slingCurves(sce_WT_cl0)[[4]]$s)
l5 <- as.data.frame(slingCurves(sce_WT_cl0)[[5]]$s)


l1 <-  l1 %>%
  add_column(lineage_names = "Endo_pt")
 
l2 <-  l2 %>%
  add_column(lineage_names = "Meso_pt")

l3 <-  l3 %>%
  add_column(lineage_names = "PGCLC_pt")

l4 <-  l4 %>%
  add_column(lineage_names = "Amnion_pt")

l5 <-  l5 %>%
  add_column(lineage_names = "Pluri_pt")


princcurves_ <- rbind(l1, l2, l3, l4, l5)

write_csv2(princcurves_, "principle_curves_lineage.csv")



l1 <- as.data.frame(rownames(as.data.frame(slingCurves(sce_WT_cl0)[[1]]$lambda)))
l2 <- as.data.frame(rownames(as.data.frame(slingCurves(sce_WT_cl0)[[2]]$lambda)))
l3 <- as.data.frame(rownames(as.data.frame(slingCurves(sce_WT_cl0)[[3]]$lambda)))
l4 <- as.data.frame(rownames(as.data.frame(slingCurves(sce_WT_cl0)[[4]]$lambda)))
l5 <- as.data.frame(rownames(as.data.frame(slingCurves(sce_WT_cl0)[[5]]$lambda)))

l1 <-  l1 %>%
  add_column(lineage_names = "Endo_pt")
  colnames(l1) <- c("barcode", "lineage_names")

l2 <-  l2 %>%
  add_column(lineage_names = "Meso_pt")
  colnames(l2) <- c("barcode", "lineage_names")


l3 <-  l3 %>%
  add_column(lineage_names = "PGCLC_pt")
  colnames(l3) <- c("barcode", "lineage_names")


l4 <-  l4 %>%
  add_column(lineage_names = "Amnion_pt")
  colnames(l4) <- c("barcode", "lineage_names")


l5 <-  l5 %>%
  add_column(lineage_names = "Pluri_pt")
  colnames(l5) <- c("barcode", "lineage_names")
  
  
barcodes_lineage <- rbind(l1, l2, l3, l4, l5)


ggplot(princ
, aes(x=umap_1, y=umap_2, colour=lineage_names)) +
  geom_point() 



meta_data = as.data.frame(WT_int_seurat@meta.data)

drop <- c("orig.ident","nCount_RNA","nFeature_RNA","batch", "louvain","pct_counts_mt", "cl0_slingPseudotime_1",
          "cl0_slingPseudotime_2", "cl0_slingPseudotime_3", "cl0_slingPseudotime_4", "cl0_slingPseudotime_5")

meta_data = meta_data[,!(names(meta_data) %in% drop)]

meta_data$barcode <- rownames(meta_data)


write_csv2(meta_data, "barcode_pseudotime_lineage.csv")



