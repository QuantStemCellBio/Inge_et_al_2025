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


sce_WT_cl0 <- slingshot(sce_WT,
                        clusterLabels = "louvain",
                        start.clus = "0",
                        end.clus = c("11","4","6","2"),
                        approx_points = 200,
                        extend = "n",
                        reducedDim = "UMAP", stretch=0)




slingLineages(sce_WT_cl0)

saveRDS(sce_WT_cl0,  "sce_WT_cl0.rds")



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

WT_int_seurat$louvain

DefaultAssay(WT_int_seurat)<- "RNA"
map(seq(1:4), ~ {
  lineage_nb <- str_c("cl0_slingPseudotime_", .x, sep = "")
  FeaturePlot(WT_int_seurat, features="EOMES") &
    scale_colour_viridis() &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[1]]$s[slingCurves(sce_WT_cl0)[[1]]$ord,]),
              aes(x = umap_1, y = umap_2)) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[2]]$s[slingCurves(sce_WT_cl0)[[2]]$ord,]),
              aes(x = umap_1, y = umap_2)) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[3]]$s[slingCurves(sce_WT_cl0)[[3]]$ord,]),
              aes(x = umap_1, y = umap_2)) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[4]]$s[slingCurves(sce_WT_cl0)[[4]]$ord,]),
              aes(x = umap_1, y = umap_2)) #&
    #geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[5]]$s[slingCurves(sce_WT_cl0)[[5]]$ord,]),
    #          aes(x = umap_1, y = umap_2)) 
  }) %>% 
  ggarrange(plotlist = ., ncol = 2, nrow = 2)


DefaultAssay(WT_int_seurat)<- "RNA"
map(rep(1:4), ~ {
  lineage_nb <- str_c("cl0_slingPseudotime_", .x, sep = "")
  FeaturePlot(WT_int_seurat, features = lineage_nb) &
    scale_colour_viridis() &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[1]]$s[slingCurves(sce_WT_cl0)[[1]]$ord,]),
              aes(x = umap_1, y = umap_2), color = "#3366ff", size = 1) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[2]]$s[slingCurves(sce_WT_cl0)[[2]]$ord,]),
              aes(x = umap_1, y = umap_2), color = "#00cc00", size = 1) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[3]]$s[slingCurves(sce_WT_cl0)[[3]]$ord,]),
              aes(x = umap_1, y = umap_2), color = "#ff9900", size = 1) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[4]]$s[slingCurves(sce_WT_cl0)[[4]]$ord,]),
              aes(x = umap_1, y = umap_2), color = "#9933ff", size = 1) #&
    #geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[5]]$s[slingCurves(sce_WT_cl0)[[5]]$ord,]),
              #aes(x = umap_1, y = umap_2), color = "#ff0099", size = 1)
}) %>%
  ggarrange(plotlist = ., ncol = 2, nrow = 2)


WT_int_seurat

# # filter low expressed genes by having 2 counts in at least 1% cells but keeping all canonical marker genes for the different lineages

all_markers <- list('HAND1','GATA3','GATA6','TLE1','EOMES','TBXT','SOX17','FOXA2')



## filter low expressed genes by having 2 counts in at least 1% cells but keeping all canonical marker genes for the different lineages
counts <- as.matrix(GetAssayData(WT_int_seurat, assay = "RNA", slot = "counts")) 

dim(counts) # we keep 12240 genes

counts <- counts %>% 
   as.data.frame() %>% 
   filter(rowSums(counts > 1) > ncol(counts)/100 | rownames(counts) %in% all_markers$marker) %>% 
   as.matrix()

dim(counts) # we keep 12240 genes

saveRDS(counts,  "counts_WT_int_seurat.rds")

pseudotimes <- slingPseudotime(sce_WT_cl0, na = FALSE)
cellweights <- slingCurveWeights(sce_WT_cl0)

set.seed(5)
BPPARAM <- BiocParallel::bpparam()
BPPARAM$workers <- 10

icMat <- evaluateK(counts = counts,
                                     pseudotime = pseudotimes,
                                       cellWeights = cellweights,
                                       #conditions = cond,
                                        parallel = TRUE,
                                        BPPARAM = BPPARAM,
                                       k = 3:15,
                                       nGenes = 300)


