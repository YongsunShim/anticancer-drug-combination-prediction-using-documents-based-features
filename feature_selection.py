import numpy as np 
import pandas as pd 
import os,sys,random



def filter_drug(original_list, drug_filter_type):
    if drug_filter_type == "target":
        drug_targets = pd.read_csv("./data/drug_data/drug_targets.csv")
        drug_list_with_targets = drug_targets['nsc_id'].unique().tolist()
        selected_drugs = list(set(original_list) & set(drug_list_with_targets))
    elif drug_filter_type == None:
        selected_drugs = original_list

    return selected_drugs


def filter_cell(original_list, cell_filter_type, available_cancer_specific_cell_list):
    if cell_filter_type == 'all':
        selected_cells = original_list
    elif cell_filter_type == 'TNBC':
        selected_cells = available_cancer_specific_cell_list['TNBC']
    else:
        selected_cells = [cell_filter_type]

    return selected_cells


def filter_cell_features(cell_data_dicts, selected_cells, cell_feats, cell_feat_filter, integrate=False):
    
    def filter_by_variance():
        if len(cell_feats) > 1:
            # if using multiple features, choose expression 
            temp = cell_data_dicts['exp']
        else:
            temp = cell_data_dicts[cell_feats[0]]
        var_df = temp.var(axis=1)
        selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)

        return selected_genes

    def filter_by_cancer_genes():
        kegg_gene_list = pd.read_csv("./data/pathway_data/kegg_gene_list.csv")
        # cancer pathway: hsa05200
        selected_genes = kegg_gene_list[kegg_gene_list['pathway']=="hsa05200"]['eg'].values.tolist()

        return selected_genes

    def filter_by_target_genes():
        drug_targets = pd.read_csv("./data/drug_data/drug_targets.csv")
        selected_genes = drug_targets['entrez'].unique().tolist()

        return selected_genes


    function_mapping = {'variance':'filter_by_variance', 'cancer':'filter_by_cancer_genes',
                        'target':'filter_by_target_genes'}
    
    
    if cell_feat_filter is not None:
        # select genes based on some criterion
        selected_genes = locals()[function_mapping[cell_feat_filter]]()
    else:
        # use all genes
        selected_genes = list(cell_data_dicts['exp'].index)
        
    if len(cell_feats) == 1 and "cellLine_word2vec" not in cell_feats:
        # if only one feat type
        feats = cell_data_dicts[cell_feats[0]]
        selected_cols = list(set(selected_cells) & set(list(feats.columns)))
        if cell_feats[0] != "mir" and cell_feats[0] != "cellLine_word2vec":
            selected_rows = list(set(selected_genes) & set(list(feats.index)))
        else:
            selected_rows = list(feats.index)
        feats = feats.loc[selected_rows, selected_cols]
        feats.dropna(axis=0, how='any', inplace=True)
    else:
        # if multiple cell feats are used
        feats_list = {}
        for feat_type in cell_feats:
            value = cell_data_dicts[feat_type]
            selected_cols = list(set(selected_cells) & set(list(value.columns)))
            if feat_type != "mir" and feat_type != "cellLine_word2vec":
                selected_rows = list(set(selected_genes) & set(list(value.index)))
            else:
                selected_rows = list(value.index)
            sel_feats = value.loc[selected_rows, selected_cols]
            sel_feats.dropna(axis=0, how='any',inplace=True)
            feats_list[feat_type] = sel_feats
        
        # if integrate is True, then the return value is a dataframe
        # otherwise, the return value is a dictionary of dataframe
        
        if integrate == True:
            feats = pd.concat(list(feats_list.values()))
        else:
            feats = feats_list
    
    return feats, selected_cols




def get_drug_feats_dim(drug_data_dicts, drug_feats):
    if len(drug_feats) == 1:
        if drug_feats[0] == 'monotherapy':
            dims = 3
        else:
            dims = len(list(drug_data_dicts[drug_feats[0]].values())[0])
    else:
        dims = 0
        for feat_type in drug_feats:
            if feat_type == 'monotherapy':
                dims += 3
            else:
                dims += len(list(drug_data_dicts[feat_type].values())[0])
    
    return dims