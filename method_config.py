import sys, os, json


# 'drug_integrate': whether or not integrating all drug features when constructing input data
# 'drug_indep': whether or not summing features of drug1 and drug2 (False: sum)
method_config_dict = {
    'deepsynergy_preuer_2018':{
        # Preuer K, Lewis RPI, Hochreiter S, et al. DeepSynergy: Predicting anti-cancer drug synergy with Deep Learning. Bioinformatics 2018; 34:1538–1546
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['morgan_fingerprint'], 
        'cell_feats': ['exp'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'NN',
        'cell_integrate': True, 
        'drug_integrate': True,
        'drug_indep': False
    },

    'nn_xia_2018':{
        # Xia F, Shukla M, Brettin T, et al. Predicting tumor cell line response to drug pairs with deep learning. BMC Bioinformatics 2018; 19:
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['morgan_fingerprint'], 
        'cell_feats': ['exp','mir','pro'], 
        'cell_feat_filter': None,
        'drug_feat_filter': 'target',
        'model_name': 'nn_xia_2018',
        'cell_integrate': False, 
        'drug_integrate': False,
        'drug_indep': True
    },

    'nn_kim_2020':{
        # Kim Y, Zheng S, Tang J, et al. Anti-cancer Drug Synergy Prediction in Understudied Tissues using Transfer Learning. 2020; 1–21
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['morgan_fingerprint','drug_target'], 
        'cell_feats': ['exp'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'nn_kim_2020',
        'cell_integrate': True,
        'drug_integrate': False,
        'drug_indep': True
    },

    'AuDNNsynergy_zhang_2018':{
        # Zhang T. Synergistic Drug Combination Prediction by Integrating Multi-omics Data in Deep Learning Models. arXiv 2018; 
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['morgan_fingerprint'], 
        'cell_feats': ['exp','mut','cop'], 
        'cell_feat_filter': None,
        'drug_feat_filter': 'target',
        'model_name': 'autoencoder',
        'cell_integrate': False,
        'drug_integrate': True,
        'drug_indep': False
    },

    'ERT_jeon_2018':{
        # Jeon M, Kim S, Park S, et al. In silico drug combination discovery for personalized cancer therapy. BMC Syst. Biol. 2018; 12:
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_target','monotherapy'], 
        'cell_feats': ['exp','mut','cop'], 
        'cell_feat_filter': 'cancer',
        'drug_feat_filter': 'target',
        'model_name': 'ERT',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    'XGBOOST_janizek_2018':{
        # Janizek JD, Celik S, Lee S-I. Explainable machine learning prediction of synergistic drug combinations for precision cancer medicine. bioRxiv 2018; 331769
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['morgan_fingerprint'], 
        'cell_feats': ['exp'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'XGBOOST',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    'XGBOOST_celebi_2019':{
        # Celebi R, Bear Don’t Walk O, Movva R, et al. In-silico Prediction of Synergistic Anti-Cancer Drug Combinations Using Multi-omics Data. Sci. Rep. 2019; 9:1–10
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['morgan_fingerprint','drug_target','monotherapy'], 
        'cell_feats': ['exp','mut','cop'], 
        'cell_feat_filter': 'cancer',
        'drug_feat_filter': 'target',
        'model_name': 'XGBOOST',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    'Logit_li_2020':{
        # Li J, Huo Y, Wu X, et al. Essentiality and transcriptome-enriched pathway scores predict drug-combination synergy. Biology (Basel). 2020; 9:1–18
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_target'], 
        'cell_feats': ['exp'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'LR',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    
    # only word2vec features
    'deepsynergy_preuer_2018_only_word2vec':{
        # Preuer K, Lewis RPI, Hochreiter S, et al. DeepSynergy: Predicting anti-cancer drug synergy with Deep Learning. Bioinformatics 2018; 34:1538–1546
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'NN',
        'cell_integrate': True, 
        'drug_integrate': True,
        'drug_indep': False
    },

    'nn_xia_2018_only_word2vec':{
        # Xia F, Shukla M, Brettin T, et al. Predicting tumor cell line response to drug pairs with deep learning. BMC Bioinformatics 2018; 19:
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',
        'drug_feats': ['drug_word2vec'],
        'cell_feats': ['cellLine_word2vec'],
        'cell_feat_filter': None,
        'drug_feat_filter': 'target',
        'model_name': 'nn_xia_2018_only_word2vec',
        'cell_integrate': False, 
        'drug_integrate': False,
        'drug_indep': True
    },

    'nn_kim_2020_only_word2vec':{
        # Kim Y, Zheng S, Tang J, et al. Anti-cancer Drug Synergy Prediction in Understudied Tissues using Transfer Learning. 2020; 1–21
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'nn_kim_2020_only_word2vec',
        'cell_integrate': True,
        'drug_integrate': False,
        'drug_indep': True
    },

    'AuDNNsynergy_zhang_2018_only_word2vec':{
        # Zhang T. Synergistic Drug Combination Prediction by Integrating Multi-omics Data in Deep Learning Models. arXiv 2018; 
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': None,
        'drug_feat_filter': 'target',
        'model_name': 'autoencoder_only_word2vec',
        'cell_integrate': False,
        'drug_integrate': True,
        'drug_indep': False
    },

    'ERT_jeon_2018_only_word2vec':{
        # Jeon M, Kim S, Park S, et al. In silico drug combination discovery for personalized cancer therapy. BMC Syst. Biol. 2018; 12:
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': 'cancer',
        'drug_feat_filter': 'target',
        'model_name': 'ERT',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    'XGBOOST_janizek_2018_only_word2vec':{
        # Janizek JD, Celik S, Lee S-I. Explainable machine learning prediction of synergistic drug combinations for precision cancer medicine. bioRxiv 2018; 331769
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'XGBOOST',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    'XGBOOST_celebi_2019_only_word2vec':{
        # Celebi R, Bear Don’t Walk O, Movva R, et al. In-silico Prediction of Synergistic Anti-Cancer Drug Combinations Using Multi-omics Data. Sci. Rep. 2019; 9:1–10
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': 'cancer',
        'drug_feat_filter': 'target',
        'model_name': 'XGBOOST',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    },

    'Logit_li_2020_only_word2vec':{
        # Li J, Huo Y, Wu X, et al. Essentiality and transcriptome-enriched pathway scores predict drug-combination synergy. Biology (Basel). 2020; 9:1–18
        'synergy_data': 'NCI_ALMANAC',
        'cell_data': 'NCI_60',
        'cell_list': 'all',        
        'drug_feats': ['drug_word2vec'], 
        'cell_feats': ['cellLine_word2vec'], 
        'cell_feat_filter': 'variance',
        'drug_feat_filter': 'target',
        'model_name': 'LR',
        'cell_integrate': True,
        'drug_integrate': True,
        'drug_indep': False
    }
}
