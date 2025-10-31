#! /usr/bin/env Rscript

# extract bw from bed 
# last 2411 by bpf 
# b.p.f@qq.com

suppressPackageStartupMessages(library("argparse"))
parser <- ArgumentParser(description='extract bw signal of given bed6')
parser$add_argument('--bed', type='character', help='input peak bed file')
parser$add_argument('--bw', type='character', help='input annotation bw file')
parser$add_argument('--lab', type='character', help='input annotation bw label')
parser$add_argument('-o', '--outputFile', type='character', help='output re-centered peak bed6 file')
parser$add_argument('--extLen', type="integer", default=20, help='extended nt length from 5p peak boundary used for sequence extraction (extLen*2), default: 20')
args <- parser$parse_args()

for(i in 1:length(args)){
  v.name <- names(args)[i]
  assign(v.name,args[[i]])
  message(paste0(v.name,": ",get(v.name)))
}


suppressPackageStartupMessages(library(GenomicRanges))
suppressPackageStartupMessages(library(rtracklayer))

# func.
readBW <- function(bw,chr,start,width){
  #bw <- read_bw
  isBw <- 
  if(grepl("bg$|bedgraph$",tolower(bw),perl=T )){
    bg <- rtracklayer::import.bedGraph(bw, which=GenomicRanges::GRanges(c(chr), IRanges(start = start+1, width = width))) # warning if no record found, IRanges is 1-based
  }else if(grepl("bw$|bigwig$",tolower(bw),perl=T )){
    bg <- rtracklayer::import.bw(bw, which=GenomicRanges::GRanges(c(chr), IRanges(start = start+1, width = width))) # warning if no record found, IRanges is 1-based
  }
  bg <- as.data.frame(bg)[,1:6]
  bg$start <- bg$start-1 # convert to 0-based bed: no err if no rec; start can be 0 !
  left <- bg$start
  wid <- bg$width
  score <- bg$score
  
  bg <- bg[,c("seqnames","start","end","width","score","strand")]
  # bg$width <- "X"
  colnames(bg)[4] <- "name"
  # head(bg,3)
  # bg <- bg %>% 
  #   dplyr::filter(seqnames==chr) # %>% 
  # bg <- bg[bg$seqnames==chr & bg$start<=start+width & bg$end>=start,] # rtracklayer::import.bw: GenomicRanges::GRanges
  
  if (nrow(bg)==0){
    bw.res.df <- data.frame("seqnames"=chr, "start"=start:(start+width-1), "end"=(start+1):(start+width), "name"="X", "score"=0, "strand"=".")
    return(bw.res.df)
  }
  
  # bw.res <- list()
  df <- data.frame("seqnames"=chr, "start"=start:(start+width-1), "end"=(start+1):(start+width), "name"="X", "score"=0, "strand"=".")
  for (i in 1:nrow(bg)){
    #rec <- bg[i,]
    # i <- 1
    # print(i)
    s <- left[i] # rec$start
    w <- wid[i]
    df[which(df$start==s):which(df$start==s+w-1),"score"] <- score[i] # 1-based
    # df <- data.frame("seqnames"=rec$seqnames, "start"=s:(s+w-1), "end"=(s+1):(s+w), "name"="X", "score"=rec$score, "strand"=rec$strand)
    # bw.res[[i]] <- df
  }
  # bw.res.df <- do.call(rbind,bw.res)
  return(df)
}
#


## test
#bw <- "/BioII/lulab_b/baopengfei/shared_reference/RMBase3/human.hg38.m6A.result.tx.bw"
#bed <- "test.bed"
#outputFile <- "./test.m6A"
# lab <- "m6A"

## test2
# bw <- "test/TCGA-W5-AA2X-01A-11R-A41D-13_mirna_gdc_realn_startEnds5.+.bedgraph"
# bed <- "/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/output/TCGA-CHOL_small/call_peak_all/cfpeakCNN_by_sample/b5_d50_p1/TCGA-W5-AA2X-01A-11R-A41D-13_mirna_gdc_realn_ext10_recenter.bed"
# outputFile <- "test/TCGA-W5-AA2X-01A-11R-A41D-13_mirna_gdc_realn_ext10.cov5"
# extLen <- 10
# lab <- "cov5"

#peak <- as.data.frame(data.table::fread(bed,data.table = F,sep = '\t',check.names = F,stringsAsFactors = F))
peak <- rtracklayer::import.bed(bed)

#tmp <- rtracklayer::import.bw(bw)
sigList <- list()
sigList <- lapply(1:length(peak), FUN = function(i) { tmp <- readBW(bw = bw, chr = seqnames(peak)[i], start = peak@ranges@start[i], width = peak@ranges@width[i]) ; 
                              return(tmp$score)
                        } 
                  )
sigDf <- as.data.frame(do.call(rbind, sigList))
#sig <- readBW(bw = bw, chr = "13022", start = 20, width = 60)
#sig <- rtracklayer::import.bw(bw, which=peak)
colnames(sigDf) <- paste0(lab,"_",1:(2*extLen)) #gsub("V",lab,colnames(sigDf))
sigDf$name <- peak$name
sigDf <- sigDf[,c(ncol(sigDf),1:(ncol(sigDf)-1))]

write.table( sigDf, outputFile, sep = '\t', row.names = F, col.names = T, quote = F )
#
