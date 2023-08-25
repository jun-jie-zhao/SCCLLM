# coding:utf-8
import pickle
import faiss
import torch
import heapq
import numpy as np
import Levenshtein
from nlgeval import compute_metrics
from tqdm import tqdm

from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

from bert_whitening import sents_to_vecs, transform_and_normalize

dim = 256

df = pd.read_csv("data/train_base_function.csv", header=None)
train_code_list = df[0].tolist()
df = pd.read_csv("data/train_ast.csv", header=None)
train_ast_list = df[0].tolist()
df = pd.read_csv("data/train_base_comment.csv", header=None)
train_nl_list = df[0].tolist()
df = pd.read_csv("data/test_base_function.csv", header=None)
test_code_list = df[0].tolist()
df = pd.read_csv("data/test_ast.csv", header=None)
test_ast_list = df[0].tolist()
df = pd.read_csv("data/test_base_comment.csv", header=None)
test_nl_list = df[0].tolist()

tokenizer = RobertaTokenizer.from_pretrained("D:\\codebert-base") #Local model file
model = RobertaModel.from_pretrained("D:\\codebert-base")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

class Retrieval(object):
    def __init__(self):
        f = open('model/code_vector_whitening.pkl', 'rb')
        self.bert_vec = pickle.load(f)
        f.close()
        f = open('model/kernel.pkl', 'rb')
        self.kernel = pickle.load(f)
        f.close()
        f = open('model/bias.pkl', 'rb')
        self.bias = pickle.load(f)
        f.close()

        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        for i in range(len(train_code_list)):
            all_texts.append(train_code_list[i])
            all_ids.append(i)
            all_vecs.append(self.bert_vec[i].reshape(1,-1))
        all_vecs = np.concatenate(all_vecs, 0)
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, ast, topK):
        body = sents_to_vecs([code], tokenizer, model)
        body = transform_and_normalize(body, self.kernel, self.bias)
        vec = body[[0]].reshape(1, -1).astype('float32')
        _, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        sim_scores = []
        sim_nls = []
        for j in sim_idx:
            if j >= len(train_ast_list):
                print("Index error: j=", j, "len(train_ast_list)=", len(train_ast_list))
                continue
            code_score = sim_jaccard(train_code_list[j].split(), code.split())
            ast_score = Levenshtein.seqratio(str(train_ast_list[j]).split(), str(ast).split())
            sim_scores.append(0.7 * code_score + 0.3 * ast_score)
            sim_nls.append(train_nl_list[j])
        topk_idx = heapq.nlargest(topK, range(len(sim_scores)), key=sim_scores.__getitem__)
        topk_nls = [sim_nls[i] for i in topk_idx]
        topk_code= [ train_code_list[sim_idx[i]] for i in topk_idx]
        return topk_nls,topk_code

if __name__ == '__main__':
    ccgir = Retrieval()
    print("Sentences to vectors")
    ccgir.encode_file()
    print("加载索引")
    ccgir.build_index(n_list=1)
    ccgir.index.nprob = 1
    sim_nl_list, c_list, sim_score_list, nl_list ,sim_nl_codelist= [], [], [], [], []
    data_list = []
    for i in tqdm(range(len(test_code_list))):
        sim_nls,sim_code = ccgir.single_query(test_code_list[i], test_ast_list[i], topK=5)
        sim_nl_list.append(sim_nls)
        sim_nl_codelist.append(sim_code)
        nl_list.append(test_nl_list[i])

    df = pd.DataFrame(nl_list)
    df.to_csv("nl.csv", index=False,header=None)
    df = pd.DataFrame(sim_nl_codelist)
    df.to_csv("code-5.csv", index=False, header=None)
    df = pd.DataFrame(sim_nl_list)
    df.to_csv("sim-5.csv", index=False,header=None)