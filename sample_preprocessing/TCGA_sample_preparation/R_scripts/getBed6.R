#! /usr/bin/env Rscript

# recenter AlkB- (old-lib) peaks 
# last 2411 by bpf 
# b.p.f@qq.com

suppressPackageStartupMessages(library("argparse"))
parser <- ArgumentParser(description='get bed6 from peak bed file')
parser$add_argument('-i', '--inputFile', type='character', help='input peak bed file')
parser$add_argument('--bedtoolsPath', type='character', default="/BioII/lulab_b/baopengfei/biosoft", help='bedtools path, default: /BioII/lulab_b/baopengfei/biosoft')
parser$add_argument('--chrSize', type='character', default="/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/genome/hg38/chrom_sizes/transcriptome_genome_sort_uniq_newTxID", 
                    help='chrSize file path, used for shuffle random regions, contains both 11+8 tx and 23 chr. default: /BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/genome/hg38/chrom_sizes/transcriptome_genome_sort_uniq_newTxID')
parser$add_argument('-o', '--outputFile', type='character', help='output re-centered peak bed6 file')
# parser$add_argument('--minOverlap', type="integer", default=10, help='min nt length for overlap of AlkB- and AlkB+, default: 10')
# parser$add_argument('--maxOverlap', type="integer", default=65, help='max nt length for overlap of AlkB- and AlkB+ , default: 65')
# parser$add_argument('--lLen', type="integer", default=6, help='nt length that consider as same position on the left boundary (usually 2*rLen, AlkB- left boundary is on the right of AlkB+ left boundary), default: 6')
# parser$add_argument('--rLen', type="integer", default=3, help='nt length that consider as same position on the right boundary, default: 3')
parser$add_argument('--extLen', type="integer", default=20, help='extended nt length from 5p peak boundary used for sequence extraction (extLen*2), default: 20')
args <- parser$parse_args()

for(i in 1:length(args)){
  v.name <- names(args)[i]
  assign(v.name,args[[i]])
  message(paste0(v.name,": ",get(v.name)))
}


# # test
# inputFile <- "/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/output/TCGA-CHOL_small/call_peak_all/cfpeakCNN_by_sample/b5_d50_p1/TCGA-W5-AA2I-01A-32R-A41D-13_mirna_gdc_realn.bed"
# outputFile <- "./test.bed"
# extLen <- 20

readPeak <- function(x){
  # x <- files[1]
  
  tmp <- as.data.frame(data.table::fread(x,data.table = F,sep = '\t',check.names = F,stringsAsFactors = F))
  colnames(tmp) <- c("peak1_chr","peak1_start","peak1_end","peak1_name","peak1_score","peak1_strand")
  tmp$sample <- gsub(".bed","",basename(x),fixed=T)
  return(tmp)
}
# files <- Sys.glob(paths = inputFiles)

ref <- data.table::fread("/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/genome/hg38/chrom_sizes/transcriptome_genome_sort_uniq_newTxID",data.table = F, header = F, sep = '\t', check.names = F, stringsAsFactors = F)
colnames(ref) <- c("transcript_id","tx.length")
#

pairList <- lapply( inputFile, readPeak )
pairDf <- as.data.frame(do.call( rbind, pairList))

pairDf$pos <- paste0(pairDf$peak1_chr," ",pairDf$peak1_start)
length(unique(pairDf$pos))

pairDf <- pairDf[!duplicated(pairDf$pos),]
pairDf$peak1_name <- paste0(pairDf$peak1_chr,"|",pairDf$peak1_start,"|")
pairDf$peak1_start <- pairDf$peak1_start-extLen
pairDf$peak1_end <- pairDf$peak1_start+extLen*2
#summary(pairDf$peak1_start)
pairDf$txLen <- ref$tx.length[match(pairDf$peak1_chr,ref$transcript_id)]
table(pairDf$peak1_start>=0) # 
table(pairDf$peak1_end<=pairDf$txLen) #
pairDf$peak1_score <- 1

# filter exceed-chr-boundary
pairDf <- pairDf[pairDf$peak1_start>0 & pairDf$peak1_end<pairDf$txLen,]

## write in peak
#write.table(peak3,paste0( gsub("bed$|bed6$|bed12$","",inputFile1,perl = T), "pair.bed"),sep = '\t',row.names = F,col.names = F,quote = F)
write.table(pairDf[,c("peak1_chr","peak1_start","peak1_end","peak1_name","peak1_score","peak1_strand")], outputFile,sep = '\t',row.names = F,col.names = F,quote = F)

