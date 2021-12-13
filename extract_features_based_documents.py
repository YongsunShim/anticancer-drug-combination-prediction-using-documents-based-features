import pandas as pd
import numpy as np
import csv

from gensim.models.word2vec import Text8Corpus
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

from callBack import *

revised_article_path = './data/revised_documents.txt'
vector_size = 256
epochs = 200

def training_word2vec():
    sentences = Text8Corpus(revised_article_path)
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=5, min_count=1, workers=4, sg=0, epochs=epochs, compute_loss=True, callbacks=[callback()])
    model.wv.save_word2vec_format('./data/word2vec.model')

def extract_features():
    model = KeyedVectors.load_word2vec_format('./data/word2vec.model') # model load
    extract_drug_features(model)
    extract_cell_line_features(model)
    
def extract_drug_features(model):
    set_f = open("./data/ComboCompoundNames_all.txt", 'r')
    set_lines = set_f.readlines()

    drug_mapping = {}
    for i in range(0, len(set_lines)):
        split = set_lines[i][:-1].split('\t')

        if split[0] not in drug_mapping:
            drug_mapping[split[0]] = split[1]

    drug_data = {}
    for key in drug_mapping.keys():
        vector = np.zeros(vector_size)
        drug = drug_mapping.get(key).lower()
        if(drug.find('\s')):
            token_split = drug.split(' ')
            for j in range(0, len(token_split)):
                if token_split[j] in model:
                    vector = vector + model[token_split[j]]
        else:
            if drug in model:
                vector = model[drug]

        drug_data[key] = vector

    data = pd.DataFrame(drug_data)
    data.head()

    data.to_csv("./data/drug_data/drug_word2vec.csv")
    
    
def extract_cell_line_features(model):
    data = pd.read_table("./data/cell_line_data/NCI-60/data_NCI-60_exp.txt")
    data.drop(columns=['Probe id','Gene name','Chromosome','Start','End','Cytoband','RefSeq(mRNA)','RefSeq(protein)'], inplace=True)
    data.set_index("Entrez gene id", inplace=True)

    data.columns = list(map(lambda x:x.split(":")[1], list(data.columns)))

    cell_line_data = {}
    for term in data.columns:
        vector = np.zeros(vector_size)
        cell_line = term.lower()
        if(cell_line.find('\s')):
            token_split = cell_line.split(' ')
            for j in range(0, len(token_split)):
                if token_split[j] in model:
                    vector = vector + model[token_split[j]]
        else:
            if cell_line in model:
                vector = model[cell_line]
                
        cell_line_data[term] = vector

    data = pd.DataFrame(cell_line_data)
    data.head()

    data.to_csv("./data/cell_line_data/NCI-60/data_NCI-60_word2vec.csv")
    
    
    
def main():
    training_word2vec()
    extract_features()
    
    
if __name__ == "__main__":
    main()