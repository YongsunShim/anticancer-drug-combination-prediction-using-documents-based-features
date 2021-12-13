import numpy as np 
import pandas as pd 
import os,sys,random
import json



def input_synergy_data(dataset):
    '''
    Load synergy datasets. load all data in the specified dataset and revise into the same format.

    param:
        dataset: str
    '''

    function_mapping = {'NCI_ALMANAC':'process_almanac', 'ONeil':'process_oneil'}

    def process_almanac():
        data = pd.read_csv("./data/synergy_data/NCI_ALMANAC/ComboDrugGrowth_Nov2017.csv")
        # summarize 3*3 or 5*3 data into one by calculating the mean score
        summary_data = data.groupby(['SCREENER','NSC1','NSC2','CELLNAME']).agg({"SCORE":'mean',"STUDY":'count'}).reset_index().rename(columns={'SCORE':'MEAN_SCORE',
                                                                                                                                 'STUDY':'count'}).astype({'NSC1':'int32','NSC2':'int32'})
        # some experiments may fail and get NA values, drop these experiments
        summary_data = summary_data.dropna()

        # remove data from 1A center (11513 data points)
        summary_data = summary_data[summary_data['SCREENER']!="1A"]

        summary_data = summary_data[['NSC1','NSC2','CELLNAME','MEAN_SCORE']].rename(columns={'NSC1':'drug1','NSC2':'drug2','CELLNAME':'cell','MEAN_SCORE':'score'})

        # replace 'MDA-MB-231/ATCC' to 'MDA-MB-231'
        summary_data = summary_data.replace('MDA-MB-231/ATCC', 'MDA-MB-231')

        return summary_data

    
    def process_oneil():
        pass

    # use locals() to run the specified function
    data = locals()[function_mapping[dataset]]()
    return data



def input_cellline_data(dataset):
    '''
    Load cell line features. load all data in the specified dataset and revise into the same format.
    Store all kinds of cell lines features in a dictionary.

    param:
        dataset: str
    '''

    function_mapping = {'NCI_60':'process_NCI60', 'CCLE':'process_CCLE'}

    def process_NCI60():

        def load_file(postfix):
            data = pd.read_table("./data/cell_line_data/NCI-60/data_NCI-60_%s.txt" % postfix)
            if postfix != 'mir':
                data.drop(columns=['Probe id','Gene name','Chromosome','Start','End','Cytoband','RefSeq(mRNA)','RefSeq(protein)'], inplace=True)
            else:
                data.drop(columns=['Probe id','Gene name','Chromosome','Start','End',
                                        'Cytoband','RefSeq(mRNA)','RefSeq(protein)','MirBase Name',"Entrez gene id"], inplace=True)
            # use entrez gene id as index
            if postfix != "mir":
                data.set_index("Entrez gene id", inplace=True)
            else:
                data.set_index('miRNA Accession #', inplace=True)
            # remove the prefix of the column names: "BR:MDA-MB-231" to "MDA-MB-231"
            data.columns = list(map(lambda x:x.split(":")[1], list(data.columns)))
            
            return data
        
        data_dicts = {}
        # load all cell line features
        for file_type in ['exp', 'cop', 'mut', 'met', 'pro', 'mir']:
            data_dicts[ file_type ] = load_file(file_type)

        # load RNAi features
        
        # load cell_line word2vec data
        cellLine_word2vec_csv = pd.read_csv("./data/cell_line_data/NCI-60/data_NCI-60_word2vec.csv", index_col=0)
        cellLine_word2vec = pd.DataFrame(cellLine_word2vec_csv)
        data_dicts['cellLine_word2vec'] = cellLine_word2vec

        return data_dicts

    
    def process_CCLE():
        pass


    data = locals()[function_mapping[dataset]]()
    return data



def input_drug_data():
    data_dicts = {}

    # load fingerprint data
    with open("./data/drug_data/drug_fingerprint_morgan_3_256.json") as f:
        fingerprint = json.load(f)
    fingerprint = pd.DataFrame(fingerprint)
    fingerprint.columns = fingerprint.columns.astype(int)
    data_dicts['morgan_fingerprint'] = fingerprint


    # load drug target (bag of words encoding)
    drug_targets = pd.read_csv("./data/drug_data/drug_targets.csv")
    drug_mapping = dict(zip(drug_targets['nsc_id'].unique().tolist(), range(len(drug_targets['nsc_id'].unique()))))
    
    gene_mapping = dict(zip(drug_targets['GeneSymbol'].unique().tolist(), range(len(drug_targets['GeneSymbol'].unique()))))
    encoding = np.zeros((len(drug_targets['nsc_id'].unique()), len(drug_targets['GeneSymbol'].unique())))

    for _, row in drug_targets.iterrows():
        encoding[drug_mapping[row['nsc_id']], gene_mapping[row['GeneSymbol']]] = 1
    
    target_feats = dict()
    for drug, row_id in drug_mapping.items():
        target_feats[int(drug)] = encoding[row_id].tolist()
    data_dicts['drug_target'] = pd.DataFrame(target_feats)


    # load monotherapy data
    with open("./data/synergy_data/NCI_ALMANAC/monotherapy_dict.json") as f:
        monotherapy = json.load(f)
    monotherapy = pd.DataFrame(monotherapy)
    monotherapy.columns = monotherapy.columns.astype(int)
    data_dicts['monotherapy'] = monotherapy
    
    
    # load drug word2vec data
    drug_word2vec_csv = pd.read_csv("./data/drug_data/drug_word2vec.csv", index_col=0)
    drug_word2vec = pd.DataFrame(drug_word2vec_csv)
    data_dicts['drug_word2vec'] = drug_word2vec

    return data_dicts


def mapping_ids():
    pass