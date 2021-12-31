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
from itertools import combinations
import random

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_path", type=str, default=None)
    parser.add_argument("corpus_path", type=str, default=None)
    parser.add_argument("reference_path", type=str, default=None)
    parser.add_argument("output_path", type=str, default=None)
    parser.add_argument("num_sent", type=int, default=3)
    parser.add_argument("num_proc", type=int, default=10)
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    id2concept = []
    data = []

    with open(args.data_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split(' ')
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

    concept2corpus = []

    def find_concept_in_corpus(concept):
        ids = []
        for i, sent in enumerate(corpus):
            if concept in sent:
                ids.append(i)
        return ids

    with Pool(args.num_proc) as p:
        for ids in tqdm(p.imap(find_concept_in_corpus, id2concept), total=len(id2concept)):
            concept2corpus.append(ids)

    def find_cooccur_corpus_sents(concepts):

        corpus_ids = []
        for concept in concepts:
            corpus_ids.append(concept2corpus[concept2id[concept]])

        ids = [i for i in range(len(concepts))]
    
        assert len(concepts) == len(corpus_ids)
        assert len(concepts) == len(ids)

        corpus_sents = []

        for i in range(len(corpus_ids), 1, -1):
            cooccur_corpus_ids = []

            for c in combinations(ids, i):
                sent_ids = corpus_ids[c[0]]
                for j in range(1, len(c)):
                    sent_ids = list(set(sent_ids).intersection(set(corpus_ids[c[j]])))

                if len(cooccur_corpus_ids) == 0:
                    cooccur_corpus_ids = sent_ids
                else:
                    cooccur_corpus_ids = list(set(cooccur_corpus_ids).union(set(sent_ids)))

            while len(cooccur_corpus_ids) > 0 and len(corpus_sents) < args.num_sent:
                sent_id = random.choice(cooccur_corpus_ids)
                if corpus[sent_id] not in reference and corpus[sent_id] not in corpus_sents:
                    corpus_sents.append(corpus[sent_id])
                cooccur_corpus_ids.remove(sent_id)

        if len(corpus_sents) > 0:
            data_with_sent = "generate a sentence with these concepts: " + " ".join(data) + ". auxiliary sentences: " + " ".join(corpus_sents) + "\n"
        else:
            data_with_sent = "generate a sentence with these concepts: " + " ".join(data) + ". auxiliary sentences: none.\n"

        return data_with_sent


    with Pool(args.num_proc) as p, open(args.output_path, 'w') as fout:
        for data_with_sents in tqdm(p.imap(find_cooccur_corpus_sents, data), total=len(data)):
            fout.write(data_with_sents)

if __name__ == "__main__":
    main()




