DATA_PATH=/your/path/to/data
CORPUS_PATH=/your/path/to/corpus
REF_PATH=/your/path/to/reference
OUTPUT_PATH=/your/path/to/output
MODEL_PATH=/your/path/to/scorer_model

python scorer_get_topn.py \
--data_path $DATA_PATH \
--corpus_path $CORPUS_PATH \
--retrieval_ids_path $REF_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH