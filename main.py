import numpy as np 
import pandas as pd 
import os,sys,random
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import keras

import tensorflow as tf


from process_data import *
from feature_selection import *
from model import get_model
from method_config import *


SYNERGY_THRES = 10
BATCH_SIZE = 256
N_EPOCHS = 200
PATIENCE = 30

available_feat_type_list = {'NCI_60':['met','mut','cop','exp']}
available_cancer_specific_cell_list = {'NCI_60':{'TNBC':['MDA-MB-231','MDA-MB-435','BT-549','HS 578T']}}


def prepare_data():
    synergy_data = input_synergy_data(config['synergy_data'])
    print("synergy data loaded")
    cell_data_dicts = input_cellline_data(config['cell_data'])
    print("cell line feats loaded")
    drug_data_dicts = input_drug_data()
    print("drug feats loaded")

    # get full drug list and cell line list
    drug_list = synergy_data['drug1'].unique().tolist()
    cell_list = synergy_data['cell'].unique().tolist()
    
    # filtering data
    drug_list = filter_drug(drug_list, config['drug_feat_filter'])
    cell_list = filter_cell(cell_list, config['cell_list'], available_cancer_specific_cell_list[config['cell_data']])
    synergy_data = synergy_data[(synergy_data['drug1'].isin(drug_list))&(synergy_data['drug2'].isin(drug_list))&(synergy_data['cell'].isin(cell_list))]
    
    # generate matrices for cell line features
    cell_feats, selected_cells = filter_cell_features(cell_data_dicts, cell_list, config['cell_feats'], config['cell_feat_filter'], config['cell_integrate'])
    synergy_data = synergy_data[synergy_data['cell'].isin(selected_cells)]
    
    print("cell line feats filtered")
    print("\n")
    print("number of drugs:", len(drug_list))
    print("number of cells:", len(selected_cells))
    print("number of data:", synergy_data.shape)
    print("\n")
    
    if config['cell_integrate'] == True:
        # in this case, cell_fets stores a dataframe containing features
        X_cell = np.zeros((synergy_data.shape[0], cell_feats.shape[0]))
        for i in tqdm(range(synergy_data.shape[0])):
            row = synergy_data.iloc[i]
            X_cell[i,:] = cell_feats[row['cell']].values
    else:
        X_cell = {}
        for feat_type in config['cell_feats']:
            print(feat_type, cell_feats[feat_type].shape[0])
            temp_cell = np.zeros((synergy_data.shape[0], cell_feats[feat_type].shape[0]))
            for i in tqdm(range(synergy_data.shape[0])):
                row = synergy_data.iloc[i]
                temp_cell[i,:] = cell_feats[feat_type][row['cell']].values
            X_cell[feat_type] = temp_cell
    
    if config['cell_integrate'] == True:
        print("cell features: ", X_cell.shape)
    else:
        print("cell features:", list(X_cell.keys()))


    # generate matrices for drug features
    # first generate individual data matrices for drug1 and drug2 and different feat types
    print("\ngenerating drug feats...")
    drug_mat_dict = {}
    for feat_type in config['drug_feats']:
        if feat_type != "monotherapy" and feat_type != "drug_word2vec":
            dim = drug_data_dicts[feat_type].shape[0]
            temp_X_1 = np.zeros((synergy_data.shape[0], dim))
            temp_X_2 = np.zeros((synergy_data.shape[0], dim))
            for i in tqdm(range(synergy_data.shape[0])):
                row = synergy_data.iloc[i]
                temp_X_1[i,:] = drug_data_dicts[feat_type][int(row['drug1'])]
                temp_X_2[i,:] = drug_data_dicts[feat_type][int(row['drug2'])]
        elif feat_type == "drug_word2vec":
            dim = drug_data_dicts[feat_type].shape[0]
            temp_X_1 = np.zeros((synergy_data.shape[0], dim))
            temp_X_2 = np.zeros((synergy_data.shape[0], dim))
            for i in tqdm(range(synergy_data.shape[0])):
                row = synergy_data.iloc[i]
                temp_X_1[i,:] = drug_data_dicts[feat_type][str(row['drug1'])]
                temp_X_2[i,:] = drug_data_dicts[feat_type][str(row['drug2'])]
        else:
            dim = 3
            temp_X_1 = np.zeros((synergy_data.shape[0], dim))
            temp_X_2 = np.zeros((synergy_data.shape[0], dim))
            for i in tqdm(range(synergy_data.shape[0])):
                row = synergy_data.iloc[i]
                temp_X_1[i,:] = drug_data_dicts[feat_type].loc[row['cell'], int(row['drug1'])]
                temp_X_2[i,:] = drug_data_dicts[feat_type].loc[row['cell'], int(row['drug2'])]
        drug_mat_dict[feat_type+"_1"] = temp_X_1
        drug_mat_dict[feat_type+"_2"] = temp_X_2
        
    # now aggregate drug features based on whether they should be summed 
    X_drug_temp = {}
    if config['drug_indep'] == False:
        for feat_type in config['drug_feats']:
            if feat_type != "monotherapy":
                temp_X = drug_mat_dict[feat_type+"_1"] + drug_mat_dict[feat_type+"_2"]
                X_drug_temp[feat_type] = temp_X
            else:
                X_drug_temp[feat_type+"_1"] = drug_mat_dict[feat_type+"_1"]
                X_drug_temp[feat_type+"_2"] = drug_mat_dict[feat_type+"_2"]
    else:
        X_drug_temp = drug_mat_dict
    
    # now aggregate drug features based on whether they should be integrated
    if config['drug_integrate'] == False:
        X_drug = X_drug_temp
    else:
        # in this case, drug feature is a numpy array instead of dict of arrays
        X_drug = np.concatenate(list(X_drug_temp.values()), axis=1)
    

    if config['drug_integrate'] == True:
        print("drug features: ", X_drug.shape)
    else:
        print("drug features")
        print(list(X_drug.keys()))
        for key, value in X_drug.items():
            print(key, value.shape)
    

    Y = (synergy_data['score']>SYNERGY_THRES).astype(int).values
    
    return X_cell, X_drug, Y



def training(X_cell, X_drug, Y):
    indices = np.random.permutation(Y.shape[0])
    training_idx, test_idx = indices[:int(0.8*Y.shape[0])], indices[int(0.8*Y.shape[0]):]
    #training_idx, test_idx = indices[:int(0.1*Y.shape[0])], indices[int(0.1*Y.shape[0]):int(0.15*Y.shape[0])]

    if config['model_name'] not in ["autoencoder", "autoencoder_only_word2vec"]:
        model = get_model(config['model_name'])
    else:
        model, encoders = get_model(config['model_name'])

    Y_train, Y_test = Y[training_idx], Y[test_idx]

    if config['model_name'] in ['NN','nn_xia_2018','nn_kim_2020']:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE),
                     ModelCheckpoint(filepath='best_model_%s.h5' % config['model_name'], monitor='val_loss', save_best_only=True)]

        if config['model_name'] == 'NN':
            X = np.concatenate([X_cell,X_drug], axis=1)
            X_train, X_test = X[training_idx], X[test_idx]
            _ = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=N_EPOCHS,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks)
        elif config['model_name'] == 'nn_xia_2018':
            X_train, X_test = {}, {}
            for key,value in X_cell.items():
                X_train[key] = value[training_idx]
                X_test[key] = value[test_idx]
            for key,value in X_drug.items():
                X_train[key] = value[training_idx]
                X_test[key] = value[test_idx]
            
            _ = model.fit(
                {'exp':X_train['exp'], 'mir':X_train['mir'], 'pro':X_train['pro'],
                     'drug1':X_train['morgan_fingerprint_1'], 'drug2':X_train['morgan_fingerprint_2']},
                Y_train,
                batch_size=BATCH_SIZE,
                epochs=N_EPOCHS,
                verbose=1,
                validation_split=0.1,
                callbacks=callbacks
            )
        elif config['model_name'] == 'nn_kim_2020':
            X_train, X_test = {}, {}
            X_train['exp'] = X_cell[training_idx]
            X_test['exp'] = X_cell[test_idx]
            for key,value in X_drug.items():
                X_train[key] = value[training_idx]
                X_test[key] = value[test_idx]
            
            _ = model.fit(
                {'exp':X_train['exp'], 'fingerprint_1':X_train['morgan_fingerprint_1'], 'fingerprint_2':X_train['morgan_fingerprint_2'],
                        'target_1':X_train['drug_target_1'], 'target_2':X_train['drug_target_2']},
                Y_train,
                batch_size=BATCH_SIZE,
                epochs=N_EPOCHS,
                verbose=1,
                validation_split=0.1,
                callbacks=callbacks
            )

        model = load_model('best_model_%s.h5' % config['model_name'])
    elif config['model_name'] == "autoencoder":
        model.compile(optimizer='adam', loss=['mse','mse', 'mse'])

        X_train, X_test = {}, {}
        for key,value in X_cell.items():
            X_train[key] = value[training_idx]
            X_test[key] = value[test_idx]
        X_train['morgan_fingerprint'] = X_drug[training_idx]
        X_test['morgan_fingerprint'] = X_drug[test_idx]

        _ = model.fit(
            [X_train['exp'], X_train['mut'], X_train['cop']],
            [X_train['exp'], X_train['mut'], X_train['cop']],
            batch_size=BATCH_SIZE,
            epochs=20,
            verbose=1
        )
        encoded_train_exp = encoders[0].predict(X_train['exp'])
        encoded_train_mut = encoders[1].predict(X_train['mut'])
        encoded_train_cop = encoders[2].predict(X_train['cop'])

        encoded_train_X = np.concatenate([encoded_train_exp,encoded_train_mut,encoded_train_cop,
                                            X_train['morgan_fingerprint']], axis=1)
        
        print("encoded_train_X:",encoded_train_X.shape)
        
        encoded_test_exp = encoders[0].predict(X_test['exp'])
        encoded_test_mut = encoders[1].predict(X_test['mut'])
        encoded_test_cop = encoders[2].predict(X_test['cop'])

        encoded_test_X = np.concatenate([encoded_test_exp,encoded_test_mut,encoded_test_cop,
                                            X_test['morgan_fingerprint']], axis=1)
        X_test = encoded_test_X
        
        model = get_model("autoencoder_NN")
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        _ = model.fit(encoded_train_X, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=100,
                        verbose=1)
    
    
    # only word2vec features
    elif config['model_name'] in ['nn_xia_2018_only_word2vec','nn_kim_2020_only_word2vec']:
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE),
                     ModelCheckpoint(filepath='best_model_%s.h5' % config['model_name'], monitor='val_loss', save_best_only=True)]

        if config['model_name'] == 'nn_xia_2018_only_word2vec':
            X_train, X_test = {}, {}
            X_train['cellLine_word2vec'] = X_cell['cellLine_word2vec'][training_idx]
            X_test['cellLine_word2vec'] = X_cell['cellLine_word2vec'][test_idx]

            X_train['drug_word2vec_1'] = X_drug['drug_word2vec_1'][training_idx]
            X_test['drug_word2vec_1'] = X_drug['drug_word2vec_1'][test_idx]
            
            X_train['drug_word2vec_2'] = X_drug['drug_word2vec_2'][training_idx]
            X_test['drug_word2vec_2'] = X_drug['drug_word2vec_2'][test_idx]
            
            _ = model.fit(
                {'cellLine_word2vec':X_train['cellLine_word2vec'],
                 'drug_word2vec_1':X_train['drug_word2vec_1'], 'drug_word2vec_2':X_train['drug_word2vec_2']},
                Y_train,
                batch_size=BATCH_SIZE,
                epochs=N_EPOCHS,
                verbose=1,
                validation_split=0.1,
                callbacks=callbacks
            )
        elif config['model_name'] == 'nn_kim_2020_only_word2vec':
            X_train, X_test = {}, {}
            
            X_train['cellLine_word2vec'] = X_cell[training_idx]
            X_test['cellLine_word2vec'] = X_cell[test_idx]

            X_train['drug_word2vec_1'] = X_drug['drug_word2vec_1'][training_idx]
            X_test['drug_word2vec_1'] = X_drug['drug_word2vec_1'][test_idx]
            
            X_train['drug_word2vec_2'] = X_drug['drug_word2vec_2'][training_idx]
            X_test['drug_word2vec_2'] = X_drug['drug_word2vec_2'][test_idx]

            _ = model.fit(
                {'cellLine_word2vec':X_train['cellLine_word2vec'],
                 'drug_word2vec_1':X_train['drug_word2vec_1'], 'drug_word2vec_2':X_train['drug_word2vec_2']},
                Y_train,
                batch_size=BATCH_SIZE,
                epochs=N_EPOCHS,
                verbose=1,
                validation_split=0.1,
                callbacks=callbacks
            )
            
        model = load_model('best_model_%s.h5' % config['model_name'])
    elif config['model_name'] == "autoencoder_only_word2vec":
        model.compile(optimizer='adam', loss=['mse','mse','mse'])
        
        X_train, X_test = {}, {}
        
        X_train['cellLine_word2vec'] = X_cell['cellLine_word2vec'][training_idx]
        X_test['cellLine_word2vec'] = X_cell['cellLine_word2vec'][test_idx]
        
        X_train['drug'] = X_drug[training_idx]
        X_test['drug'] = X_drug[test_idx]
        
        _ = model.fit(
            [X_train['cellLine_word2vec']],
            [X_train['cellLine_word2vec']],
            batch_size=BATCH_SIZE,
            epochs=20,
            verbose=1
        )
        encoded_train_cellLine_word2vec = encoders.predict(X_train['cellLine_word2vec'])

        encoded_train_X = np.concatenate([encoded_train_cellLine_word2vec,X_train['drug']], axis=1)
        
        print("encoded_train_X:",encoded_train_X.shape)
        
        encoded_test_cellLine_word2vec = encoders.predict(X_test['cellLine_word2vec'])

        encoded_test_X = np.concatenate([encoded_test_cellLine_word2vec,X_test['drug']], axis=1)
        X_test = encoded_test_X
        
        model = get_model("autoencoder_NN")
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        _ = model.fit(encoded_train_X, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=100,
                        verbose=1)
    else:
        X = np.concatenate([X_cell,X_drug], axis=1)
        X_train, X_test = X[training_idx], X[test_idx]
        model.fit(X_train, Y_train)

    return model, X_test, Y_test

def evaluate(model, X_test, Y_test):
    if config['model_name'] in ['NN','nn_xia_2018','nn_kim_2020',
                                'nn_xia_2018_only_word2vec','nn_kim_2020_only_word2vec'
                               ]:
        if config['model_name'] == 'NN':
            pred = model.predict(X_test)
        elif config['model_name'] == 'nn_xia_2018':
            pred = model.predict({'exp':X_test['exp'], 'mir':X_test['mir'], 'pro':X_test['pro'],
                                    'drug1':X_test['morgan_fingerprint_1'], 'drug2':X_test['morgan_fingerprint_2']})
        elif config['model_name'] == 'nn_kim_2020':
            pred = model.predict({'exp':X_test['exp'], 'fingerprint_1':X_test['morgan_fingerprint_1'], 'fingerprint_2':X_test['morgan_fingerprint_2'],
                        'target_1':X_test['drug_target_1'], 'target_2':X_test['drug_target_2']})
            
        elif config['model_name'] == 'nn_xia_2018_only_word2vec':
            pred = model.predict({'cellLine_word2vec':X_test['cellLine_word2vec'],
                                  'drug_word2vec_1':X_test['drug_word2vec_1'], 'drug_word2vec_2':X_test['drug_word2vec_2']})
        elif config['model_name'] == 'nn_kim_2020_only_word2vec':
            pred = model.predict({'cellLine_word2vec':X_test['cellLine_word2vec'],
                                  'drug_word2vec_1':X_test['drug_word2vec_1'], 'drug_word2vec_2':X_test['drug_word2vec_2']})
            
    elif config['model_name'] == "autoencoder":
        pred = model.predict(X_test)
    elif config['model_name'] == "autoencoder_only_word2vec":
        pred = model.predict(X_test)
    else:
        pred = model.predict_proba(X_test)[:,1]
    
    auc = roc_auc_score(Y_test, pred)
    ap = average_precision_score(Y_test, pred)
    f1 = f1_score(Y_test, np.round(pred))

    val_results = {'AUC':auc, 'AUPR':ap, 'f1':f1, 'Y_test':Y_test.tolist(), 'Y_pred':pred.tolist()}

    return val_results



def main(method):
    X_cell, X_drug, Y = prepare_data()
    print("data loaded")
    model, X_test, Y_test = training(X_cell, X_drug, Y)
    print("training finished")
    val_results = evaluate(model, X_test, Y_test)

    # save results
    if 'only_word2vec' in method:
        rand_num = random.randint(1,1000000)
        with open("./results/only_word2vec_result/%s_%s.json"%(method, str(rand_num)), "w") as f:
            json.dump(val_results, f)
    else:
        rand_num = random.randint(1,1000000)
        with open("./results/baseline_result/%s_%s.json"%(method, str(rand_num)), "w") as f:
            json.dump(val_results, f)
    
if __name__ == "__main__":
    method = sys.argv[1]
    config = method_config_dict[method]
    print(method, config)
    main(method)