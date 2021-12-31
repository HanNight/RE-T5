import argparse
import numpy as np
from scipy import spatial
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import json
import random
import os
import pickle
import pickle
import random
import heapq

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_path", type=str, default=None)
    parser.add_argument("corpus_path", type=str, default=None)
    parser.add_argument("retrieval_ids_path", type=str, default=None)
    parser.add_argument("output_path", type=str, default=None)
    parser.add_argument("num_sent", type=int, default=3)

    parser.add_argument("model_path", type=int, default=None)
    parser.add_argument("batch_size", type=int, default=256)
    parser.add_argument("max_length", type=int, default=64)

    parser.add_argument("device", type=str, default="cuda:0")

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    data = []

    with open(args.data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if ": " in line:
                line = line[line.find(": ") + 2:]
            line = line.split(' ')
            data.append(line)

    with open(args.corpus_path, 'r', encoding='utf-8') as fin:
        corpus = [line.strip() for line in fin]

    with open(args.retrieval_ids, 'rb') as fin:
        retrieval_ids = pickle.load(fin)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model = model.to(args.device)

    def get_topn_sents(concepts, corpus_ids, batch_size=256, max_length=64, num_sent=3):

        corpus_sents = []
        sent_scores = []
        
        num = int(len(corpus_ids) / batch_size)
        remain = len(corpus_ids) - num * batch_size

        for i in range(num):
            sent1_set = [" ".join(concepts)] * batch_size
            sent2_set = [corpus[i] for i in corpus_ids[i*batch_size:(i+1)*batch_size]]
            inputs = tokenizer(sent1_set, sent2_set, padding='max_length', max_length=max_length, truncation='only_second', return_tensors="pt")
            input_ids = inputs["input_ids"].to(args.device)
            token_type_ids = inputs["token_type_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sent_scores = sent_scores + outputs[0][:,0].cpu().tolist()
    
        if remain > 0:
            sent1_set = [" ".join(concepts)] * remain
            sent2_set = [corpus[i] for i in corpus_ids[-remain:]]
            inputs = tokenizer(sent1_set, sent2_set, padding='max_length', max_length=max_length, truncation='only_second', return_tensors="pt")
            input_ids = inputs["input_ids"].to(args.device)
            token_type_ids = inputs["token_type_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sent_scores = sent_scores + outputs[0][:,0].cpu().tolist()
    
        best_ids = heapq.nlargest(num_sent, range(len(sent_scores)), sent_scores.__getitem__)
        for idx in best_ids:
            corpus_sents.append(corpus[corpus_ids[idx]])

        if len(corpus_sents) > 0:
            concepts_with_sents = "generate a sentence with these concepts: " + " ".join(concepts) + ". auxiliary sentences: " + " ".join(corpus_sents) + "\n"
        else:
            concepts_with_sents = "generate a sentence with these concepts: " + " ".join(concepts) + ". auxiliary sentences: none.\n"

        return concepts_with_sents

    with open(output_path, 'w') as fout:
        for i in tqdm(range(len(data))):
            concepts_with_sents = get_topn_sents(data[i], retrieval_ids[i], args.batch_size, args.max_length, args.num_sent)
            fout.write(concepts_with_sents)

if __name__ == "__main__":
    main()