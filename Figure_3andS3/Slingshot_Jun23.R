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




WT_int_seurat <- readRDS(file = "seurat_converted.rds")

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$louvain

WT_int_seurat


sce_WT <- as.SingleCellExperiment(WT_int_seurat, assay = "RNA")

table(sce_WT$louvain)

FeaturePlot(object = WT_int_seurat, features ="SOX2")


sce_WT_cl0 <- slingshot(sce_WT,
                        clusterLabels = "louvain",
                        start.clus = "0",
                        end.clus = c("11","4","6","2"),
                        approx_points = 200,
                        extend = "n",
                        reducedDim = "UMAP", stretch=0.8)




slingLineages(sce_WT_cl0)


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

as.data.frame(slingCurves(sce_WT_cl0)[[1]])



DefaultAssay(WT_int_seurat)<- "RNA"
map(seq(1:4), ~ {
  lineage_nb <- str_c("cl0_slingPseudotime_", .x, sep = "")
  FeaturePlot(WT_int_seurat, features = "GATA3") &
    scale_colour_viridis() &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[1]]$s[slingCurves(sce_WT_cl0)[[1]]$ord,]),
              aes(x = umap_1, y = umap_2)) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[2]]$s[slingCurves(sce_WT_cl0)[[2]]$ord,]),
              aes(x = umap_1, y = umap_2)) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[3]]$s[slingCurves(sce_WT_cl0)[[3]]$ord,]),
              aes(x = umap_1, y = umap_2)) &
    geom_path(data = as.data.frame(slingCurves(sce_WT_cl0)[[4]]$s[slingCurves(sce_WT_cl0)[[4]]$ord,]),
              aes(x = umap_1, y = umap_2)) 
  }) %>% 
  ggarrange(plotlist = ., ncol = 2, nrow = 2)

WT_int_seurat@meta.data$cl0_slingPseudotime_1

WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Endo_lineage = if_else(cl0_slingPseudotime_1>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Meso_lineage = if_else(cl0_slingPseudotime_2>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(PGCLC_lineage = if_else(cl0_slingPseudotime_3>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Amnion_lineage = if_else(cl0_slingPseudotime_4>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Pluri_lineage = if_else(cl0_slingPseudotime_5>0, TRUE, FALSE))

#saveRDS(WT_int_seurat, file = "slingshot_all.rds")

WT_int_seurat <- readRDS(file = "slingshot_all.rds")

WT_int_seurat

WT_int_seurat <- WT_int_seurat_raw
WT_int_seurat_raw <- WT_int_seurat

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$batch
WT_int_seurat <- subset(WT_int_seurat, idents=3)



Idents(WT_int_seurat) <- WT_int_seurat@meta.data$Endo_lineage
Endo_lineage <- subset(WT_int_seurat, idents="TRUE")

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$Meso_lineage
Meso_lineage <- subset(WT_int_seurat, idents="TRUE")

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$Amnion_lineage
Amnion_lineage <- subset(WT_int_seurat, idents="TRUE")

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$PGCLC_lineage
PGCLC_lineage <- subset(WT_int_seurat, idents="TRUE")



plotting_dynamics_pt <- function(goi){
  Endo_data <- FeatureScatter(
    Endo_lineage,
    "Endo_pt",
    goi)
  
  
  Endo_data <- Endo_data$data
  Endo_data$Trajectory <- "Endoderm_Trajectory" 
  Endo_data$Pseudotime <- Endo_data$Endo_pt
  Endo_data <- subset(Endo_data, select = -c(Endo_pt, colors))
  
  Meso_data <- FeatureScatter(
    Meso_lineage,
    "Meso_pt",
    goi)
  
  
  Meso_data <- Meso_data$data
  Meso_data$Trajectory <- "Mesoderm_Trajectory" 
  Meso_data$Pseudotime <- Meso_data$Meso_pt
  Meso_data <- subset(Meso_data, select = -c(Meso_pt, colors))
  
  PGCLC_data <- FeatureScatter(
    PGCLC_lineage,
    "PGCLC_pt",
    goi)
  
  
  PGCLC_data <- PGCLC_data$data
  PGCLC_data$Trajectory <- "PGCLC_Trajectory" 
  PGCLC_data$Pseudotime <- PGCLC_data$PGCLC_pt
  PGCLC_data <- subset(PGCLC_data, select = -c(PGCLC_pt, colors))
  PGCLC_data
  
  Amnion_data <- FeatureScatter(
    Amnion_lineage,
    "Amnion_pt",
    goi)
  
  Amnion_data <- Amnion_data$data
  Amnion_data$Trajectory <- "Amnion_Trajectory" 
  Amnion_data$Pseudotime <- Amnion_data$Amnion_pt
  Amnion_data <- subset(Amnion_data, select = -c(Amnion_pt, colors))
  Amnion_data
  
  output <- Meso_data %>% full_join(Endo_data)
  output <- output %>% full_join(Amnion_data)
  output <- output %>% full_join(PGCLC_data)
  
  
  colnames(output)[1] = "Expression"
  ggplot(output, aes(x=Pseudotime , y=Expression, color=Trajectory)) +
    theme_minimal() + scale_fill_manual(values=c(ingeo_colours[[3]], ingeo_colours[[1]], ingeo_colours[[5]], ingeo_colours[[4]])) +
    scale_color_manual(values=c(ingeo_colours[[3]], ingeo_colours[[1]], ingeo_colours[[5]], ingeo_colours[[4]] )) + geom_smooth() +
    ggtitle(goi) +
    theme(plot.title = element_text(hjust = 0.5))
  
}


plotting_dynamics_pt('GDF3')


plotting_dynamics_EM_pt <- function(goi){
  Endo_data <- FeatureScatter(
    Endo_lineage,
    "Endo_pt",
    goi)
  
  
  Endo_data <- Endo_data$data
  Endo_data$Trajectory <- "Endoderm_Trajectory" 
  Endo_data$Pseudotime <- Endo_data$Endo_pt
  Endo_data <- subset(Endo_data, select = -c(Endo_pt, colors))
  
  Meso_data <- FeatureScatter(
    Meso_lineage,
    "Meso_pt",
    goi)
  
  
  Meso_data <- Meso_data$data
  Meso_data$Trajectory <- "Mesoderm_Trajectory" 
  Meso_data$Pseudotime <- Meso_data$Meso_pt
  Meso_data <- subset(Meso_data, select = -c(Meso_pt, colors))
  
  
  output <- Meso_data %>% full_join(Endo_data)
  
  
  colnames(output)[1] = "Expression"
  ggplot(output, aes(x=Pseudotime , y=Expression, color=Trajectory)) + theme_minimal() + scale_fill_manual(values=c(ingeo_colours[[1]], ingeo_colours[[5]])) +
    scale_color_manual(values=c(ingeo_colours[[1]], ingeo_colours[[5]])) + geom_smooth() +
    ggtitle(goi) +
    theme(plot.title = element_text(hjust = 0.5))
  
}

plotting_dynamics_EM_pt('TBXT')

pdf("TBXT.pdf", width=6, height=4)
plotting_dynamics_EM_pt('TBXT')
dev.off();

FeaturePlot(object = WT_int_seurat, features ="MESP1")

