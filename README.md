# RE-T5
Repo for paper ["Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression"](https://aclanthology.org/2021.findings-acl.269/)(ACL-IJCNLP 2021).

<center><img src="https://user-images.githubusercontent.com/36069169/147724625-90e1a993-b291-416b-956a-55bbabed3f69.png" width="300px"></center>

## Prepare Data

### CommonGen Dataset
You can get CommonGen dataset from [their official website](https://inklab.usc.edu/CommonGen/) or [Huggingface Datasets Hub](https://huggingface.co/datasets/viewer/?dataset=common_gen)

### External Corpora
We provide our `external corpora` consisting of [VATEX](https://arxiv.org/abs/1904.03493), [Activity](https://arxiv.org/abs/1705.00754), [SNLI](https://aclanthology.org/D15-1075/), and [MNLI](https://aclanthology.org/N18-1101/) in `data` folder.
In addition, you can build your own external corpora.

## Pretrain

### Prepare Pre-training Data
1. Download Entity Vocab of ConceptNet [here](https://drive.google.com/file/d/17666iO8HNd-nDE0CUCDuuz4l4FR2a4Tb/view?usp=sharing).
2. Use the external corpora to construct a pre-training dataset similar to the CommonGen format following the example:
```shell
DATA_PATH=/your/path/to/data
OUTPUT_PATH=/your/path/to/output
CONCEPT_PATH=/your/path/to/concept
TEST_SRC_PATH=/your/path/to/commongen_test_src_path
TEST_TGT_PATH=/your/path/to/train/commongen_test_tgt_path

python data_augmentation.py \
--data_path $DATA_PATH\
--output_path $OUTPUT_PATH\
--concept_path $CONCEPT_PATH\
--commongen_test_src_path $TEST_SRC_PATH \
--commongen_test_tgt_path $TEST_TGT_PATH
```
3. Use `Matching Retriever` to retrieve prototype sentences for pre-training data following the example: 
```shell
DATA_PATH=/your/path/to/data
CORPUS_PATH=/your/path/to/corpus
REF_PATH=/your/path/to/reference
OUTPUT_PATH=/your/path/to/output

python matching_retriever.py \
--data_path $DATA_PATH \
--corpus_path $CORPUS_PATH \
--reference_path $REF_PATH \
--output_path $OUTPUT_PATH
```

### Pretrain the RE-T5
You can pretrain RE-T5 following the example:
```shell
export PYTHONPATH="../":"${PYTHONPATH}"

DATA_DIR=/your/path/to/data
OUTPUT_DIR=/your/path/to/output
MODEL_NAME=t5-base
DATASET_MODE=shuffle # shuffle or original

python finetune.py \
--gpus=4 \
--gradient_accumulation_steps=4 \
--warmup_steps=5000 \
--data_dir=$DATA_DIR \
--model_name_or_path=$MODEL_NAME \
--num_train_epochs=5 \
--learning_rate=2e-6 \
--weight_decay=0.01 \
--adam_epsilon=1e-6 \
--train_batch_size=16 \
--eval_batch_size=16 \
--max_source_length=256 \
--max_target_length=256 \
--val_max_target_length=256 \
--test_max_target_length=256 \
--val_check_interval=1.0 --n_val=-1 \
--output_dir=$OUTPUT_DIR \
--do_train \
--eval_beam=5 \
--overwrite_output_dir \
--dataset_mode=$DATASET_MODE \
 "$@"
``` 

**Note**: `DATASET_MODE` is used to determine whether to shuffle the input concepts.

## Train

### Train a Trainable Retriever
1. Prepare the training and validation set. For each concept set in CommonGen training set, we use its paired sentence as a positive example and we randomly sample another sentence, also from the training set, as a negative example. The format of data json file should follow the following format:
```json
{"sentence1": "chicken cheese pizza broccoli", "sentence2": "A grilled pizza with chicken, broccoli and cheese.", "label": 1}
{"sentence1": "chicken cheese pizza broccoli", "sentence2": "The pan pizza topped with broccoli, chicken, and cheese, is ready for the oven", "label": 1}
{"sentence1": "chicken cheese pizza broccoli", "sentence2": "model released portrait of an old boy opening the film character presents in holidays", "label": 0}
```
2. Train the `Trainable Retriever` following the example:
```shell
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
```

### Use `Trainable Retriever` to Retrieve Prototype Sentences for CommonGen Data
1. Obtain sentences that contained the concepts.
```shell
DATA_PATH=/your/path/to/data
CORPUS_PATH=/your/path/to/corpus
REF_PATH=/your/path/to/reference
OUTPUT_PATH=/your/path/to/output

python retrieve_all_ids.py \
--data_path $DATA_PATH \
--corpus_path $CORPUS_PATH \
--reference_path $REF_PATH \
--output_path $OUTPUT_PATH
```
2. Use the Trainable Retriever to rank these candidates from previous step to select top-k prototype sentences.
```shell
DATA_PATH=/your/path/to/data
CORPUS_PATH=/your/path/to/corpus
RE_IDS_PATH=/your/path/to/retrieval_ids/from/previous/step
OUTPUT_PATH=/your/path/to/output
MODEL_PATH=/your/path/to/trainable_model

python trainable_retriever.py \
--data_path $DATA_PATH \
--corpus_path $CORPUS_PATH \
--retrieval_ids_path $RE_IDS_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH
```

### Train the RE-T5
You can train RE-T5 following the example:
```shell
export PYTHONPATH="../":"${PYTHONPATH}"

DATA_DIR=/your/path/to/data
OUTPUT_DIR=/your/path/to/output
MODEL_PATH=/your/path/to/model
DATASET_MODE=shuffle # shuffle or original

python finetune.py \
--gpus=4 \
--gradient_accumulation_steps=3 \
--warmup_steps=400 \
--data_dir=$DATA_DIR \
--model_name_or_path=$MODEL_PATH \
--num_train_epochs=20 \
--learning_rate=5e-5 \
--train_batch_size=64 \
--eval_batch_size=64 \
--max_source_length=128 \
--max_target_length=32 \
--val_check_interval=1.0 --n_val=-1 \
--output_dir=$OUTPUT_DIR \
--do_train \
--eval_beam=5 \
--overwrite_output_dir \
--dataset_mode=$DATASET_MODE \
 "$@"
```

## Inference and Evaluation

### Inference
Follow the example:
```shell
MODEL_PATH=/your/path/to/model
TEST_SRC_PATH=/your/path/to/test_source
TEST_TGT_PATH=/your/path/to/test_target
OUTPUT_PATH=/your/path/to/output
SCORE_PATH=/your/path/to/rouge.json

python run_eval.py $MODEL_PATH \
    $TEST_SRC_PATH \
    $OUTPUT_PATH \
    --reference_path $TEST_TGT_PATH \
    --score_path $SCORE_PATH \
    --task summarization \
    --max_source_length 128 \
    --max_target_length 32 \
    --num_beams 5 \
    --length_penalty 0.6 \
    --early_stopping true \
    --bs 32
```

**Note**: I have provided our model inference output in `results` folder for reference.

### Evaluation
Please follow [CommonGen GitHub Repo](https://github.com/INK-USC/CommonGen) to build the evaluation environment and evaluate the results.

**Note**: Use `clean_out.py` on your results before evaluation. 

## Citation
```bibtex
@inproceedings{wang-etal-2021-retrieval-enhanced,
    title = "Retrieval Enhanced Model for Commonsense Generation",
    author = "Wang, Han  and
      Liu, Yang  and
      Zhu, Chenguang  and
      Shou, Linjun  and
      Gong, Ming  and
      Xu, Yichong  and
      Zeng, Michael",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.269",
    doi = "10.18653/v1/2021.findings-acl.269",
    pages = "3056--3062",
}
```