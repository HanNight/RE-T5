TRAIN_DATA_PATH=/your/path/to/train/data
VAL_DATA_PATH=/your/path/to/val/data
MODEL_NAME=bert-base-cased
OUTPUT_DIR=/your/path/to/output

python -m torch.distributed.launch \
    --nproc_per_node 2 run_glue.py \
    --model_name_or_path ${MODEL_NAME} \
    --train_file ${TRAIN_DATA_PATH} \
    --validation_file ${VAL_DATA_PATH} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 64 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ${OUTPUT_DIR} \
    --save_total_limit 5

# python -m torch.distributed.launch \
#     --nproc_per_node 2 run_glue.py \
#     --model_name_or_path bert-base-cased \
#     --train_file /raid/t-hanwang/commongen/retrieval_scorer/sents_regression/data/train.json \
#     --validation_file /raid/t-hanwang/commongen/retrieval_scorer/sents_regression/data/val.json \
#     --do_train \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 64 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 3.0 \
#     --output_dir /raid/t-hanwang/commongen/retrieval_scorer/sents_regression/bert_base_scorer \
#     --save_total_limit 5