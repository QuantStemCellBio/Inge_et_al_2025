library(tidyverse)
library(Seurat)
library(SingleCellExperiment)
library(ggplot2)
library(ggpubr)
library(viridis)
library(tradeSeq)
library(glue)
library(slingshot)

##CSVs
assoRes <- read_csv("assoRes.csv")
startRes <- read_csv("startRes.csv")
endRes <- read_csv("endRes.csv")

##FDR functions
padjust_comparisons_startres <- function(df){
  df <- df %>% mutate(padj_lineage1 = p.adjust(pvalue_lineage1, "fdr"), .after = pvalue_lineage1) %>%
   mutate(padj_lineage2 = p.adjust(pvalue_lineage2, "fdr"), .after = pvalue_lineage2) %>%
    mutate(padj_lineage3 = p.adjust(pvalue_lineage3, "fdr"), .after = pvalue_lineage3) %>%
    mutate(padj_lineage4 = p.adjust(pvalue_lineage4, "fdr"), .after = pvalue_lineage4) %>%
    mutate(padj_lineage5 = p.adjust(pvalue_lineage5, "fdr"), .after = pvalue_lineage5)
  
  return(df)
  
  
  
  
}
padjust_comparisons_assoRes <- function(df){
  df <- df %>% mutate(padj_1 = p.adjust(pvalue_1, "fdr"), .after = pvalue_1) %>%
    mutate(padj_2 = p.adjust(pvalue_2, "fdr"), .after = pvalue_2) %>%
    mutate(padj_3 = p.adjust(pvalue_3, "fdr"), .after = pvalue_3) %>%
    mutate(padj_4 = p.adjust(pvalue_4, "fdr"), .after = pvalue_4) %>%
    mutate(padj_5 = p.adjust(pvalue_5, "fdr"), .after = pvalue_5)
  
  return(df)
  
  
  
  
}
padjust_comparisons <- function(df){
  df <- df %>% mutate(padj_1vs2 = p.adjust(pvalue_1vs2, "fdr"), .after = pvalue_1vs2) %>%
    mutate(padj_1vs3 = p.adjust(pvalue_1vs3, "fdr"), .after = pvalue_1vs3) %>%
    mutate(padj_1vs4 = p.adjust(pvalue_1vs4, "fdr"), .after = pvalue_1vs4) %>%
    mutate(padj_1vs5 = p.adjust(pvalue_1vs5, "fdr"), .after = pvalue_1vs5) %>%
    mutate(padj_2vs3 = p.adjust(pvalue_2vs3, "fdr"), .after = pvalue_2vs3) %>%
    mutate(padj_2vs4 = p.adjust(pvalue_2vs4, "fdr"), .after = pvalue_2vs4) %>%
    mutate(padj_2vs5 = p.adjust(pvalue_2vs5, "fdr"), .after = pvalue_2vs5) %>%
    mutate(padj_3vs4 = p.adjust(pvalue_3vs4, "fdr"), .after = pvalue_3vs4) %>%
    mutate(padj_3vs5 = p.adjust(pvalue_3vs5, "fdr"), .after = pvalue_3vs5) %>%
    mutate(padj_4vs5 = p.adjust(pvalue_4vs5, "fdr"), .after = pvalue_4vs5)
  
  
  return(df)
  
  
  
  
}

startRes <- padjust_comparisons_startres(startRes)
endRes <- padjust_comparisons(endRes)
assoRes <- padjust_comparisons_assoRes(assoRes)

##AssoRes
assoRes_endo <- assoRes %>% filter(padj_1 < 0.05) %>% arrange(desc(waldStat_1))
assoRes_endo_s <- assoRes_endo %>% filter(!(`...1` %in% assoRes_meso$...1))


assoRes_meso <- assoRes %>% filter(padj_2 < 0.05) %>% arrange(desc(waldStat_2))
assoRes_meso_s <- assoRes_meso %>% filter(!(`...1` %in% assoRes_endo$...1))

assoRes_meso_s


##EndoRes

endRes1v2 <- endRes %>% filter(padj_1vs2 < 0.05) %>% arrange(desc(waldStat_1vs2))

