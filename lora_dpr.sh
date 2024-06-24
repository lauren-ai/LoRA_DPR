MODEL_NAME="lora"
OUTPUT_DIR=results/$MODEL_NAME
TOTAL_STEPS=80000
TOTAL_BATCH_SIZE=10  
TOTAL_BATCH_SIZE=$((TOTAL_BATCH_SIZE/2))  # one text generates two spans: 'anchor', 'contextual_span'
BATCH_SIZE_PER_GPU=10   # 每个batch的样本数,max=55
GRAD_ACCU=1

init_weitht=./Cotmae_Retriever/
CUDA_VISIBLE_DEVICES=0 python run_pretraining_copy.py \
        --model_name_or_path $init_weitht \
        --output_dir $OUTPUT_DIR/model \
        --do_train \
        --logging_steps 20 \
        --save_steps 100000 \
        --save_total_limit 4 \
        --fp16 \
        --logging_dir $OUTPUT_DIR/tfboard/$MODEL_NAME \
        --warmup_ratio 0.1 \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRAD_ACCU \
        --learning_rate 1e-4 \
        --max_steps $TOTAL_STEPS \
        --overwrite_output_dir \
        --dataloader_drop_last \
        --dataloader_num_workers 16 \
        --max_seq_length 128 \
        --train_path data/lora_data \
        --weight_decay 0.01 \
        --data_type mixed \
        --encoder_mask_ratio 0.30 \
        --decoder_mask_ratio 0.45 \
        --use_decoder_head \
        --enable_head_mlm \
        --n_head_layers 2 \
        --num_train_epochs 3
