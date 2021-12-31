import argparse
import numpy as np
from scipy import spatial
# import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
import json
import random
import os
import pickle
import spacy
import copy
from nltk.stem import WordNetLemmatizer

class ConceptGenerator:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.pipeline = [("tagger", self.nlp.tagger), ("parser", self.nlp.parser)]

    #TODO : can generate concept shuffling ????
    def check_availability(self, sentence):
        def check_availability_sentence(x):
            # x = x.numpy().decode('utf-8')
            # doc = self.nlp(str(x))
            doc = self.nlp(x)
            V_concepts = []
            N_concepts = []
            original_tokens = []
            for token in doc:
                original_tokens.append(token.text_with_ws)
                if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                    V_concepts.append(token.text_with_ws)
            for noun_chunk in doc.noun_chunks:
                root_noun = noun_chunk[-1]
                if root_noun.pos_ == "NOUN":
                    N_concepts.append(root_noun.text_with_ws)
            if len(N_concepts) >= 2 or len(V_concepts) >= 2:
                if len(set(N_concepts)) == 1 or len(set(V_concepts)) == 1:
                    return False
                else:
                    return True
            else:
                return False

        # result = tf.py_function(check_availability_sentence, [sentence['text']], [tf.bool])[0]
        result = check_availability_sentence(sentence)
        return result

    def cor_generate(self, prompt):
        # doc = self.nlp(str(prompt))
        doc = self.nlp(prompt)
        V_concepts = []
        N_concepts = []
        original_tokens = []
        for token in doc:
            original_tokens.append(token.text_with_ws)
            if token.pos_.startswith('V') and token.is_alpha and not token.is_stop:
                V_concepts.append(token.text_with_ws)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                N_concepts.append(root_noun.text_with_ws)

        if len(N_concepts) >= 2:
            previous = copy.deepcopy(N_concepts)
            while previous == N_concepts:
                random.shuffle(N_concepts)
        if len(V_concepts) >= 2:
            previous = copy.deepcopy(V_concepts)
            while previous == V_concepts:
                random.shuffle(V_concepts)

        shuffled_tokens = []
        N_concepts_index = 0
        V_concepts_index = 0
        for tok in original_tokens:
            if tok in V_concepts and V_concepts_index < len(V_concepts):
                shuffled_tokens.append(V_concepts[V_concepts_index])
                V_concepts_index += 1
            elif tok in N_concepts and N_concepts_index < len(N_concepts):
                shuffled_tokens.append(N_concepts[N_concepts_index])
                N_concepts_index += 1
            else:
                shuffled_tokens.append(tok)

        assert len(shuffled_tokens) == len(original_tokens)

        result = ''.join([token for token in shuffled_tokens])
        return result

    def c2s_generate(self, prompt):
        # doc = self.nlp(str(prompt))
        doc = self.nlp(prompt)

        matched_concepts = []
        for token in doc:
            if (token.pos_.startswith('V') or token.pos_.startswith('PROP')) and token.is_alpha and not token.is_stop:
                matched_concepts.append(token.lemma_)
        for noun_chunk in doc.noun_chunks:
            root_noun = noun_chunk[-1]
            if root_noun.pos_ == "NOUN":
                matched_concepts.append(root_noun.lemma_)

        result = " ".join([token for token in matched_concepts])
        return result

    def generate(self, prompt):

        negative_sampling = random.uniform(0,1) < 0.5
        if negative_sampling:
            return self.cor_generate(prompt)
        else:
            return self.c2s_generate(prompt)

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data_path", type=str, default=None)
    parser.add_argument("output_path", type=str, default=None)
    parser.add_argument("concept_path", type=str, default=None)
    parser.add_argument("commongen_test_src_path", type=str, default=None)
    parser.add_argument("commongen_test_tgt_path", type=str, default=None)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open(args.concept_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    with open(args.commongen_test_src_path, "r", encoding="utf-8") as fin:
        test_src = [set(line.strip().split(" ")) for line in fin]
    
    with open(args.commongen_test_tgt_path, "r", encoding="utf-8") as fin:
        test_tgt = [line.strip() for line in fin]

    with open(args.data_path, 'r', encoding='utf-8') as fin:
        data = [line.strip() for line in fin]

    lemmatizer = WordNetLemmatizer()

    generator = ConceptGenerator()

    with open(args.output_path, 'w') as fout:
        for sent in tqdm(data):
            if sent in test_tgt:
                continue

            if generator.check_availability(sent):
                concepts = generator.c2s_generate(sent)
                concepts = concepts.split(" ")

                if len(concepts) < 3:
                    continue

                remove_concepts = []
                for i in range(len(concepts)):
                    if concepts[i] in id2concept:
                        continue
                    else:
                        concept_lemma = lemmatizer.lemmatize(concepts[i])
                        if concept_lemma in id2concept:
                            concepts[i] = concept_lemma
                        else:
                            remove_concepts.append(concepts[i])

                if len(concepts) - len(remove_concepts) < 3:
                    continue

                for remove_concept in remove_concepts:
                    concepts.remove(remove_concept)

                while len(concepts) > 5:
                    concepts.remove(random.choice(concepts))

                if set(concepts) not in test_src:
                    fout.write(" ".join(concepts) + "\t" + sent + "\n")

if __name__ == "__main__":
    main()
        

            
    