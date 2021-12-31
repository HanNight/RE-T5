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
from itertools import combinations
import pickle
import random

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_path", type=str, default=None)
    parser.add_argument("corpus_path", type=str, default=None)
    parser.add_argument("reference_path", type=str, default=None)
    parser.add_argument("output_path", type=str, default=None)
    parser.add_argument("num_proc", type=int, default=10)
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    id2concept = []
    data = []

    with open(args.data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if ": " in line:
                line = line[line.find(": ") + 2:]
            line = line.split(' ')
            data.append(line)
            for cp in line:
                if cp not in id2concept:
                    id2concept.append(cp)

    concept2id = {w: i for i, w in enumerate(id2concept)}

    with open(args.corpus_path, 'r', encoding='utf-8') as fin:
        corpus = [line.strip() for line in fin]

    reference = {}

    with open(args.reference_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            reference[line] = 1

    def find_concept_in_corpus(concept):
        ids = []
        for i, sent in enumerate(corpus):
            if concept in sent and sent not in reference:
                ids.append(i)
        return ids

    concept2corpus = []

    with Pool(args.num_proc) as p:
        for ids in tqdm(p.imap(find_concept_in_corpus, id2concept), total=len(id2concept)):
            concept2corpus.append(ids)

    def find_cooccur_corpus_ids(concepts):

        in_corpus_ids = []
        for concept in concepts:
            in_corpus_ids.append(concept2corpus[concept2id[concept]])

        ids = [i for i in range(len(data))]
    
        assert len(concepts) == len(in_corpus_ids)
        assert len(concepts) == len(ids)

        cooccur_corpus_ids=[]

        for i in range(len(in_corpus_ids), 2, -1):

            for c in combinations(ids, i):
                corpus_ids = in_corpus_ids[c[0]]
                for j in range(1, len(c)):
                    corpus_ids = list(set(corpus_ids).intersection(set(in_corpus_ids[c[j]])))

                if len(cooccur_corpus_ids) == 0:
                    cooccur_corpus_ids = corpus_ids
                else:
                    cooccur_corpus_ids = list(set(cooccur_corpus_ids).union(set(corpus_ids)))
        
        return cooccur_corpus_ids

    outputs = []
    with Pool(args.num_proc) as p, open(args.output_path, 'wb') as fout:
        for cooccur_corpus_ids in tqdm(p.imap(find_cooccur_corpus_ids, data), total=len(data)):
            outputs.append(cooccur_corpus_ids)
        pickle.dump(outputs, fout)

if __name__ == "__main__":
    main()