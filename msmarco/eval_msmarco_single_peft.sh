BASE_DIR=$(dirname "$PWD")
MODEL_NAME=$1
MODEL_TYPE=bert

LR=2e-5
SEED=42
SPLITS=2  # 1个GPU
TRAIN_N_PASSAGES=16

MODEL_PATH=$BASE_DIR/results/$MODEL_NAME/model      
OUTPUT_DIR=$BASE_DIR/results/$MODEL_NAME/tevatron
huggdataset=/media/guest/DATA/zhlhong/huggingface/datasets/

set -e
set -x
CUDA_VISIBLE_DEVICES=1 python -m tevatron_peft.driver.train \
        --model_name_or_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR/retriever_model_s1 \
        --save_strategy no \
        --train_dir marco/$MODEL_TYPE/train \
        --fp16 \
        --per_device_train_batch_size 36 \
        --learning_rate $LR \
        --num_train_epochs 3 \
        --train_n_passages $TRAIN_N_PASSAGES \
        --seed $SEED \
        --dataloader_num_workers 2 \
        --cache_dir $huggdataset
       --grad_cache \
       --gc_q_chunk_size 32 \
       --gc_p_chunk_size 16

bash test_retriever_copy_peft.sh $OUTPUT_DIR/retriever_model_s1 $OUTPUT_DIR s1_
