import numpy as np
import os
import gensim
import pandas as pd
import csv
import functools as ft
import nltk
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
from run_readJson import load_dataset

final_simcore=[]

nltk.download('stopwords')
nltk.download('punkt')

PATH_TO_WORD2VEC = os.path.expanduser("./data/word2vec/GoogleNews-vectors-negative300.bin")
# PATH_TO_GLOVE = os.path.expanduser("./data/glove/glove.6B.300d.txt")

PATH_TO_FREQUENCIES_FILE = "./data/sentence_similarity/frequencies.tsv"
PATH_TO_DOC_FREQUENCIES_FILE = "./data/sentence_similarity/doc_frequencies.tsv"

###data
data_str = "./data/test_qa.json"
mid_file_path = "./data/mid_test_qa.csv"
final_file_path = "./data/final_test_qa.csv"
final_score_path="./data/final_score_qa.csv"
dataset=load_dataset(data_str,mid_file_path,final_file_path)

###Preparation
STOP = set(nltk.corpus.stopwords.words("english"))
class Sentence:
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        self.tokens = [t.lower() for t in nltk.word_tokenize(normalized_sentence)]
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]

word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True)
# tmp_file = "./data/glove/glove.6B.300d.w2v.txt"
# glove2word2vec(PATH_TO_GLOVE, tmp_file) #To load Glove, we have to convert the downloaded GloVe file to word2vec format and then load the embeddings into a Gensim model. This will take some time.
# glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

def read_tsv(f):
    frequencies = {}
    with open(f) as tsv:
        tsv_reader = csv.reader(tsv, delimiter="\t")
        for row in tsv_reader:
            frequencies[row[0]] = int(row[1])

    return frequencies
frequencies = read_tsv(PATH_TO_FREQUENCIES_FILE)
doc_frequencies = read_tsv(PATH_TO_DOC_FREQUENCIES_FILE)
doc_frequencies["NUM_DOCS"] = 1288431  #in order to compute weighted averages of word embeddings later, we are going to load a file with word frequencies. These word frequencies have been collected from Wikipedia and saved in a tab-separated file.

###Similarity methods
def run_avg_benchmark(df=None, model=None, use_stoplist=False, doc_freqs=None):
    sentences1 = [Sentence(s) for s in df['question1']]
    sentences2 = [Sentence(s) for s in df['question2']]
    if doc_freqs is not None:
        N = doc_freqs["NUM_DOCS"]
    sims = []
    num=0
    for (sent1, sent2) in zip(sentences1, sentences2):
        tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
        tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

        tokens1 = [token for token in tokens1 if token in model]
        tokens2 = [token for token in tokens2 if token in model]

        if len(tokens1) == 0 or len(tokens2) == 0:
            sims.append(0)
            continue

        tokfreqs1 = Counter(tokens1)
        tokfreqs2 = Counter(tokens2)

        weights1 = [tokfreqs1[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                    for token in tokfreqs1] if doc_freqs else None
        weights2 = [tokfreqs2[token] * math.log(N / (doc_freqs.get(token, 0) + 1))
                    for token in tokfreqs2] if doc_freqs else None

        embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
        embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

        sim = cosine_similarity(embedding1, embedding2)[0][0]
        sims.append(sim)

        if sim >= 0.9:
            if df["answer1"][num]==df["answer2"][num]:
                final_simcore.append((df["video_id"][num], df["id1"][num], df["id2"][num], df["question1"][num],
                                      df["question2"][num], df["answer1"][num], df["answer2"][num],sim,1))
            else:
                final_simcore.append((df["video_id"][num], df["id1"][num], df["id2"][num], df["question1"][num],
                                      df["question2"][num], df["answer1"][num], df["answer2"][num], sim, 0))
        num=num+1

    final = pd.DataFrame(final_simcore,
                         columns=["video_id", "id1", "id2", "question1", "question2", "answer1", "answer2", "sim",
                                  "isEqual"])
    final.to_csv(final_score_path, index=False)

if __name__ == '__main__':
    # benchmarks = [("AVG-W2V", ft.partial(run_avg_benchmark, model=word2vec, use_stoplist=False)),
    #               ("AVG-GLOVE", ft.partial(run_avg_benchmark, model=glove, use_stoplist=False))]
    # benchmarks = [("AVG-W2V", ft.partial(run_avg_benchmark,df=dataset, model=word2vec, use_stoplist=False))]
    run_avg_benchmark(dataset,word2vec,False,None)
