#!/bin/bash
BASE_DIR=$(dirname "$PWD")

datename=$(date +%Y%m%d%H%M%S)
MODEL_PATH=$1
OUTPUT_DIR=$2
PREFIX=$3
SPLITS=0
ENCODE_TEMP_PATH=${OUTPUT_DIR}/ENCODE_TEMP_${datename}
huggdataset=/media/guest/DATA/zhlhong/huggingface/datasets/

set -e
set -x

######################################
#Encode corpus
######################################
mkdir -p $ENCODE_TEMP_PATH/encoding/corpus-s4/
for i in $(seq -f "%02g" 0 7)
do
  touch $ENCODE_TEMP_PATH/encoding/corpus-s4/split$i.pt
  CUDA_VISIBLE_DEVICES=$SPLITS python -m tevatron_peft.driver.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --fp16 \
  --per_device_eval_batch_size 128 \
  --cache_dir $huggdataset \
  --encode_in_path marco/bert/corpus/split$i.json \
  --encoded_save_path $ENCODE_TEMP_PATH/encoding/corpus-s4/split$i.pt
done

######################################
#Encode query
######################################
mkdir -p $ENCODE_TEMP_PATH/encoding/query-s4/
touch $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt
CUDA_VISIBLE_DEVICES=$SPLITS python -m tevatron_peft.driver.encode \
  --output_dir None \
  --model_name_or_path $MODEL_PATH \
  --fp16 \
  --q_max_len 32 \
  --encode_is_qry \
  --per_device_eval_batch_size 128 \
  --cache_dir $huggdataset \
  --encode_in_path marco/bert/query/dev.query.json \
  --encoded_save_path $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt

######################################
#Faiss retrieve
######################################
mkdir -p $ENCODE_TEMP_PATH/dev_ranks
for i in $(seq -f "%02g" 0 7)
do
  touch $ENCODE_TEMP_PATH/dev_ranks/$i
  #sudo chmod 777 $ENCODE_TEMP_PATH/dev_ranks/$i
  CUDA_VISIBLE_DEVICES=$SPLITS python -m tevatron_peft.faiss_retriever \
        --query_reps $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt \
        --passage_reps $ENCODE_TEMP_PATH/encoding/corpus-s4/split$i.pt \
        --depth 1000 \
        --batch_size -1 \
        --save_ranking_to $ENCODE_TEMP_PATH/dev_ranks/$i
done

######################################
#Faiss Reduce
######################################
touch $ENCODE_TEMP_PATH/dev.rank.tsv
python -m tevatron_peft.faiss_retriever.reducer \
  --score_dir $ENCODE_TEMP_PATH/dev_ranks \
  --query $ENCODE_TEMP_PATH/encoding/query-s4/qry.pt \
  --save_ranking_to $ENCODE_TEMP_PATH/dev.rank.tsv


######################################
#Score
######################################
touch $ENCODE_TEMP_PATH/dev.rank.tsv
python score_to_marco.py $ENCODE_TEMP_PATH/dev.rank.tsv

touch $ENCODE_TEMP_PATH/dev.rank.tsv.marco
python msmarco_eval.py \
  --path_to_reference=marco/qrels.dev.tsv \
  --path_to_candidate=$ENCODE_TEMP_PATH/dev.rank.tsv.marco \
  --save_folder=$OUTPUT_DIR \
  --prefix=$PREFIX

mv $ENCODE_TEMP_PATH/dev.rank.tsv $OUTPUT_DIR/${PREFIX}dev.rank.tsv
rm -rf $ENCODE_TEMP_PATH
