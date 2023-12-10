#! /bin/bash
export TRANSFORMERS_CACHE="/cm/archive/namlh35/.cache"
export HF_DATASETS_CACHE="/cm/archive/namlh35/.cache"
export https_proxy=http://10.16.29.10:8080
export WANDB_DISABLED="true"
export TMPDIR=/cm/archive/namlh35/tmp

LANGUAGE=pharo
LANGUAGE_SRC=/cm/archive/namlh35/code-comment-classification/processed_data/novalid/$LANGUAGE

# for TYPE in deprecation expand ownership pointer rational summary usage
# for TYPE in developmentnotes expand parameters summary usage
# for TYPE in classreferences collaborators example intent keyimplementationpoints keymessages responsibilities
for TYPE in classreferences collaborators example intent keyimplementationpoints keymessages responsibilities
do
    CUDA_VISIBLE_DEVICES=5,6 python3 training/run.py \
    --seed 0 \
    --model_short_name codebert \
    --model_name_or_path /cm/archive/namlh35/code-comment-classification/results/post_pretraining_codebert2/processed_data_all \
    --train_file $LANGUAGE_SRC/$TYPE/train.csv \
    --validation_file $LANGUAGE_SRC/$TYPE/valid.csv \
    --test_file $LANGUAGE_SRC/$TYPE/test.csv \
    --output_dir ./results/selection_with_valid/codebert-post_pretrained \
    --text_column_names class,comment_sentence \
    --label_column_name instance_type \
    --metric_for_best_model f1 \
    --metric_name f1 \
    --text_column_delimiter "</s>" \
    --max_seq_length 64 \
    --max_query_length 64 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --num_train_epochs 20 \
    --do_train \
    --do_predict \
    --load_best_model_at_end \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --overwrite_output_dir
done
# --cross_validation \
# --extra_file ./data/llama-13b-chat-classification-processed.json \