library(tidyverse)
library(Seurat)
library(SingleCellExperiment)
library(ggplot2)
library(ggpubr)
library(viridis)
library(tradeSeq)
library(glue)
library(slingshot)


sce <- readRDS(file = "All_lineages_fitGAM.rds")
counts <- readRDS(file = "counts_WT_int_seurat.rds")
sce_WT_cl0 <- readRDS(file = "sce_WT_cl0.rds")
WT_int_seurat <- readRDS(file = "FS_seurat_converted_raw_count.rds")

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




table(rowData(sce)$tradeSeq$converged)


#Association test...
#Which genes are associated with the meso timecourse but not the endo timecourse?
assoRes <- associationTest(sce, global=TRUE,
                                                               lineages = TRUE,
                                                               l2fc = log2(1.3))



head(assoRes)

write.csv(assoRes, "assoRes.csv", row.names=TRUE)



#Start_End_analysis_comparison
startRes <- startVsEndTest(sce, global=TRUE,
                                                                lineages = TRUE,
                                                                 l2fc = log2(1.3))

head(startRes)
write.csv(startRes, "startRes.csv", row.names=TRUE)


oStart <- order(startRes$waldStat, decreasing = TRUE)

sigGeneStart <- names(sce)[oStart[3]]

plotSmoothers(sce, counts, gene = sigGeneStart)

 endRes_fitgam_wt <- diffEndTest(models = sce,
                                 global=TRUE,
                                 pairwise = TRUE,
                                l2fc = log2(1.3))
 
 
write.csv(startRes, "endRes.csv", row.names=TRUE)


patternRes_fitgam_wt <- patternTest(models = sce,
                                     global=TRUE,
                                     pairwise = TRUE,
                                     l2fc = log2(1.3))



write.csv(patternRes_fitgam_wt, "Pattern_Between_lineage.csv", row.names=TRUE)


apply(slingClusterLabels(sce_WT_cl0), 1, which.max)




png("plot.png"):

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

dev.off();


earlyDERes_fitgam_wt <- earlyDETest(models = sce,
                                     global=TRUE,
                                     pairwise = TRUE,
                                     knots = c(4, 5),
                                     l2fc = log2(1.3))



write.csv(earlyDERes_fitgam_wt, "earlyDERes_fitgam_wt_k4_5.csv", row.names=TRUE)



earlyDERes_fitgam_wt <- earlyDETest(models = sce,
                                    global=TRUE,
                                    pairwise = TRUE,
                                    knots = c(4, 6),
                                    l2fc = log2(1.3))



write.csv(earlyDERes_fitgam_wt, "earlyDERes_fitgam_wt_k4_6.csv", row.names=TRUE)


earlyDERes_fitgam_wt <- earlyDETest(models = sce,
                                    global=TRUE,
                                    pairwise = TRUE,
                                    knots = c(4, 7),
                                    l2fc = log2(1.3))



write.csv(earlyDERes_fitgam_wt, "earlyDERes_fitgam_wt_k4_7.csv", row.names=TRUE)



earlyDERes_fitgam_wt <- earlyDETest(models = sce,
                                    global=TRUE,
                                    pairwise = TRUE,
                                    knots = c(4, 8),
                                    l2fc = log2(1.3))



write.csv(earlyDERes_fitgam_wt, "earlyDERes_fitgam_wt_k4_8.csv", row.names=TRUE)






colnames(earlyDERes_fitgam_wt)

earlyDERes_fitgam_wt[order(earlyDERes_fitgam_wt$pvalue_1vs2),]

sig = earlyDERes_fitgam_wt %>% filter(pvalue_1vs2 < 0.05)


sig[order(sig$waldStat_1vs2, -sig$pvalue_1vs2),]


plotting_dynamics_EM_pt <- function(goi, max_, min_){
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
  ggplot(output, aes(x=Pseudotime , y=log(Expression), color=Trajectory)) +
    geom_point(alpha = 0) + theme_minimal() + scale_fill_manual(values=c(ingeo_colours[[1]], ingeo_colours[[5]])) +
    scale_color_manual(values=c(ingeo_colours[[1]], ingeo_colours[[5]])) + geom_smooth() +
    ggtitle(goi) +
    theme(plot.title = element_text(hjust = 0.5)) 
  
}


WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Endo_lineage = if_else(cl0_slingPseudotime_1>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Meso_lineage = if_else(cl0_slingPseudotime_2>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(PGCLC_lineage = if_else(cl0_slingPseudotime_3>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Amnion_lineage = if_else(cl0_slingPseudotime_4>0, TRUE, FALSE))
WT_int_seurat@meta.data = WT_int_seurat@meta.data %>% mutate(Pluri_lineage = if_else(cl0_slingPseudotime_5>0, TRUE, FALSE))


Idents(WT_int_seurat) <- WT_int_seurat@meta.data$batch
WT_int_seurat <- subset(WT_int_seurat, idents=3)


Idents(WT_int_seurat) <- WT_int_seurat@meta.data$Endo_lineage
Endo_lineage <- subset(WT_int_seurat, idents="TRUE")

Idents(WT_int_seurat) <- WT_int_seurat@meta.data$Meso_lineage
Meso_lineage <- subset(WT_int_seurat, idents="TRUE")


earlyDERes_fitgam_wt[order(-earlyDERes_fitgam_wt$waldStat_1vs3),]

plotting_dynamics_EM_pt('GATA3')
