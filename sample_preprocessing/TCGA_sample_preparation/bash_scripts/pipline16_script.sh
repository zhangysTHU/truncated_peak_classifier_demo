cd /BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev

dst="TCGA_small16" # snc_pandora_hsa # GSE71008_NCpool  TCGA-CHOL_small
dedup=_all
pre=/BioII/lulab_b/baopengfei/projects/WCHSU-FTC
chrSize=/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/genome/hg38/chrom_sizes/transcriptome_genome_sort_uniq_newTxID
chrFa=$pre/exSeek-dev/genome/hg38/fasta_newTxID/combine19.fa
inDir=/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/output/${dst}/call_peak${dedup}/cfpeakCNN_by_sample/b5_d50_p1
bwDir=/BioII/lulab_b/baopengfei/projects/WCHSU-FTC/output/${dst}/call_peak${dedup}/tbigwig_RNA_EM
modDir=/BioII/lulab_b/baopengfei/shared_reference/RMBase3
outDir=/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/output
CORES=1
extLen=20

source /BioII/lulab_b/baopengfei/mambaforge/bin/activate py37

# 创建输出目录
mkdir -p ${outDir}/recentered_peak_bed
mkdir -p ${outDir}/fasta
mkdir -p ${outDir}/sequence_seconderyStructure
# mkdir -p ${outDir}/one_hot
mkdir -p ${outDir}/rna_mod_annotation
# mkdir -p ${outDir}/transcriptome_bigwig
mkdir -p ${outDir}/coverage

## get re-centered peak bed (unpaired TCGA)
for i1 in `cat /BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/sample_txts_21/sample_ids_16.txt`
do
echo $i1
Rscript /BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/scripts/AlkB/getBed6.R \
  -i ${inDir}/${i1}.bed \
  -o ${outDir}/recentered_peak_bed/${i1}_ext${extLen}_recenter.bed \
  --extLen ${extLen}
done

## peak region fasta, k-mer, and 2nd structure
for i1 in `cat /BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/sample_txts_21/sample_ids_16.txt`
do
echo $i1

#fasta
bedtools getfasta -s -nameOnly -fi $chrFa -bed ${outDir}/recentered_peak_bed/${i1}_ext${extLen}_recenter.bed \
 | sed s/"(-)"/""/g  | sed s/"(+)"/""/g \
 > ${outDir}/fasta/${i1}_ext${extLen}_recenter.fa

#2nd structure
python3 ./scripts/AlkB/rnafold_parallel.py  ${outDir}/fasta/${i1}_ext${extLen}_recenter.fa 1 1234 ${outDir}/sequence_seconderyStructure/${i1}_ext${extLen}_recenter.fa.csv ${CORES}

# #one-hot
# ./scripts/AlkB/one_hot_encode.sh <(cut -d "," -f 5 ${outDir}/secondary_structure/${i1}_ext${extLen}_recenter.fa.csv) > ${outDir}/one_hot/${i1}_ext${extLen}_recenter.fa.onehot

#RNAmod annotation
for i in m1A m5C m6A m7G Nm otherMod Pseudo RNA-editing
do
echo $i
Rscript /BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/scripts/AlkB/extractBwFromBed.R \
--bed ${outDir}/recentered_peak_bed/${i1}_ext${extLen}_recenter.bed \
--bw ${modDir}/human.hg38.${i}.result.tx.bw \
--lab ${i} --extLen ${extLen} \
-o ${outDir}/rna_mod_annotation/${i1}_ext${extLen}_recenter.${i}
done
done

## sample tx bw feature (max-min norm)
for i1 in `cat /BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/sample_txts_21/sample_ids_16.txt`
do
echo $i1
Rscript /BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/scripts/AlkB/extractBwFromBed.R \
--bed ${outDir}/recentered_peak_bed/${i1}_ext${extLen}_recenter.bed \
--bw ${bwDir}/${i1}.transcriptome.bigWig \
--lab cov --extLen ${extLen} \
-o ${outDir}/coverage/${i1}_ext${extLen}.cov
done

# ## get 5' 3' coverage
# for i1 in `cat /BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/sample_txts_21/sample_ids_16.txt`
# do
# echo $i1
# ./scripts/AlkB/getBamRead53BoundaryCov.py \
# -i ../output/$dst/tbam/${i1}/bam-EM/merge19_sort/merged.sorted.bam \
# -m both --input-strandness forward \
# -o ${outDir}/coverage/${i1}
# Rscript /BioII/lulab_b/baopengfei/projects/WCHSU-FTC/exSeek-dev/scripts/AlkB/extractBwFromBed.R \
# --bed ${outDir}/recentered_peak_bed/${i1}_ext${extLen}_recenter.bed \
# --bw ${outDir}/coverage/${i1}_startEnds5.+.bedgraph \
# --lab cov5 --extLen ${extLen} \
# -o ${outDir}/coverage/${i1}_ext${extLen}.cov5
# done

date