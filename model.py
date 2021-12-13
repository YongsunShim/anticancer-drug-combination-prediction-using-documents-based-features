import numpy as np 
import pandas as pd 
import os,sys,random

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from bisect import bisect_right, bisect_left
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.regularizers import l2, l1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import scale


def get_model(model_name):

    def SVM():
        model = SVC(gamma='auto', probability=True)
        return model

    def LR():
        model = LogisticRegression()
        return model

    def XGBOOST():
        model = GradientBoostingClassifier(max_features='auto')
        return model

    def RF():
        model = RandomForestClassifier()
        return model
    
    def ERT():
        model = ExtraTreesClassifier()
        return model

    def NN():
        dropout = 0.5
        batchNorm = True
        num_classes = 1

        model = Sequential()
        model.add(Dense(1024))
        if batchNorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(512))
        if batchNorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation='sigmoid'))
        return model
    
    def nn_xia_2018():
        # input shape numbers should be decided by code in the future
        input_exp = keras.Input(shape=(18077,), name='exp')
        input_mir = keras.Input(shape=(680,), name='mir')
        input_pro = keras.Input(shape=(162,), name='pro')
        input_drug1 = keras.Input(shape=(256,), name='drug1')
        input_drug2 = keras.Input(shape=(256,), name='drug2')

        exp = layers.Dense(2048, activation='relu')(input_exp)
        exp = layers.Dense(1024, activation='relu')(exp)
        exp = layers.Dense(512, activation='relu')(exp)

        mir = layers.Dense(1024, activation='relu')(input_mir)
        mir = layers.Dense(512, activation='relu')(mir)
        mir = layers.Dense(256, activation='relu')(mir)

        pro = layers.Dense(1024, activation='relu')(input_pro)
        pro = layers.Dense(512, activation='relu')(pro)
        pro = layers.Dense(256, activation='relu')(pro)

        shared_drug_encoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(128, activation='relu')
            ]
        )

        drug1 = shared_drug_encoder(input_drug1)
        drug2 = shared_drug_encoder(input_drug2)

        combined = layers.concatenate([exp, mir, pro, drug1, drug2])
        x = layers.Dense(2048, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        pred = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(
            inputs = [input_exp, input_mir, input_pro, input_drug1, input_drug2],
            outputs = pred
        )

        return model

    def nn_kim_2020():
        input_exp = keras.Input(shape=(480,), name='exp')
        input_fingerprint_1 = keras.Input(shape=(256,), name='fingerprint_1')
        input_fingerprint_2 = keras.Input(shape=(256,), name='fingerprint_2')
        input_target_1 = keras.Input(shape=(545,), name='target_1')
        input_target_2 = keras.Input(shape=(545,), name='target_2')

        exp = layers.Dense(1024, activation='relu')(input_exp)
        exp = layers.Dense(512, activation='relu')(exp)
        exp = layers.Dense(256, activation='relu')(exp)

        fingerprint_1 = layers.Dense(128, activation='relu')(input_fingerprint_1)
        target_1 = layers.Dense(128, activation='relu')(input_target_1)
        combined_1 = layers.concatenate([fingerprint_1, target_1])
        combined_1 = layers.Dense(512, activation='relu')(combined_1)
        combined_1 = layers.Dense(128, activation='relu')(combined_1)

        fingerprint_2 = layers.Dense(128, activation='relu')(input_fingerprint_2)
        target_2 = layers.Dense(128, activation='relu')(input_target_2)
        combined_2 = layers.concatenate([fingerprint_2, target_2])
        combined_2 = layers.Dense(512, activation='relu')(combined_2)
        combined_2 = layers.Dense(128, activation='relu')(combined_2)

        combined = layers.concatenate([combined_1, exp, combined_2])
        combined = layers.Dense(1024, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        pred = layers.Dense(1, activation='sigmoid')(combined)

        model = keras.Model(
            inputs = [input_exp, input_fingerprint_1, input_fingerprint_2, input_target_1, input_target_2],
            outputs = pred
        )

        return model
    

    def autoencoder():
        input_exp = keras.Input(shape=(18077,), name='exp')
        input_mut = keras.Input(shape=(8930,), name='mut')
        input_cop = keras.Input(shape=(17844,), name='cop')

        # exp auto-encoder
        exp = layers.Dense(2048, activation='relu')(input_exp)
        exp = layers.Dense(1024, activation='relu')(exp)
        exp = layers.Dense(512, activation='relu')(exp)
        exp_encoder_output = layers.Dense(256, activation='relu')(exp)

        exp_encoder = keras.Model(input_exp, exp_encoder_output)

        exp = layers.Dense(512, activation='relu')(exp_encoder_output)
        exp = layers.Dense(1024, activation='relu')(exp)
        exp = layers.Dense(2048, activation='relu')(exp)
        exp_decoder_output = layers.Dense(18077, activation='relu')(exp)

        exp_autoencoder = keras.Model(input_exp, exp_decoder_output)


        # mut auto-encoder
        mut = layers.Dense(2048, activation='relu')(input_mut)
        mut = layers.Dense(1024, activation='relu')(mut)
        mut = layers.Dense(512, activation='relu')(mut)
        mut_encoder_output = layers.Dense(256, activation='relu')(mut)

        mut_encoder = keras.Model(input_mut, mut_encoder_output)

        mut = layers.Dense(512, activation='relu')(mut_encoder_output)
        mut = layers.Dense(1024, activation='relu')(mut)
        mut = layers.Dense(2048, activation='relu')(mut)
        mut_decoder_output = layers.Dense(8930, activation='relu')(mut)

        mut_autoencoder = keras.Model(input_mut, mut_decoder_output)


        # cop auto-encoder
        cop = layers.Dense(2048, activation='relu')(input_cop)
        cop = layers.Dense(1024, activation='relu')(cop)
        cop = layers.Dense(512, activation='relu')(cop)
        cop_encoder_output = layers.Dense(256, activation='relu')(cop)

        cop_encoder = keras.Model(input_cop, cop_encoder_output)

        cop = layers.Dense(512, activation='relu')(cop_encoder_output)
        cop = layers.Dense(1024, activation='relu')(cop)
        cop = layers.Dense(2048, activation='relu')(cop)
        cop_decoder_output = layers.Dense(17844, activation='relu')(cop)

        cop_autoencoder = keras.Model(input_cop, cop_decoder_output)

        autoencoder = keras.Model(
            inputs = [input_exp, input_mut, input_cop],
            outputs = [exp_decoder_output, mut_decoder_output, cop_decoder_output]
        )
        #keras.utils.plot_model(autoencoder, "autoencoder.png", show_shapes=True)

        return autoencoder, (exp_encoder, mut_encoder, cop_encoder)
    
    def autoencoder_NN():
        dropout = 0.5
        batchNorm = True
        num_classes = 1

        model = Sequential()
        model.add(Dense(2048))
        if batchNorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(1024))
        if batchNorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(512))
        if batchNorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        model.add(Dense(num_classes, activation='sigmoid'))
        return model
    
    
    # Only word2vec features
    def nn_xia_2018_only_word2vec():
        # input shape numbers should be decided by code in the future
        input_cellLine_word2vec = keras.Input(shape=(256,), name='cellLine_word2vec')
        
        input_drug_word2vec_1 = keras.Input(shape=(256,), name='drug_word2vec_1')
        input_drug_word2vec_2 = keras.Input(shape=(256,), name='drug_word2vec_2')

        cellLine_word2vec = layers.Dense(1024, activation='relu')(input_cellLine_word2vec)
        cellLine_word2vec = layers.Dense(512, activation='relu')(cellLine_word2vec)
        cellLine_word2vec = layers.Dense(256, activation='relu')(cellLine_word2vec)

        shared_drug_encoder = keras.Sequential(
            [
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(128, activation='relu')
            ]
        )

        drug_word2vec_1 = shared_drug_encoder(input_drug_word2vec_1)
        drug_word2vec_2 = shared_drug_encoder(input_drug_word2vec_2)
        

        combined = layers.concatenate([cellLine_word2vec, drug_word2vec_1, drug_word2vec_2])
        x = layers.Dense(2048, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        pred = layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(
            inputs = [input_cellLine_word2vec, 
                      input_drug_word2vec_1, input_drug_word2vec_2],
            outputs = pred
        )

        return model


    def nn_kim_2020_only_word2vec():
        input_cellLine_word2vec = keras.Input(shape=(256,), name='cellLine_word2vec')
        
        input_drug_word2vec_1 = keras.Input(shape=(256,), name='drug_word2vec_1')
        input_drug_word2vec_2 = keras.Input(shape=(256,), name='drug_word2vec_2')

        cellLine_word2vec = layers.Dense(1024, activation='relu')(input_cellLine_word2vec)
        cellLine_word2vec = layers.Dense(512, activation='relu')(cellLine_word2vec)
        cellLine_word2vec = layers.Dense(256, activation='relu')(cellLine_word2vec)

        combined_1 = layers.Dense(512, activation='relu')(input_drug_word2vec_1)
        combined_1 = layers.Dense(128, activation='relu')(combined_1)

        combined_2 = layers.Dense(512, activation='relu')(input_drug_word2vec_2)
        combined_2 = layers.Dense(128, activation='relu')(combined_2)

        combined = layers.concatenate([combined_1, cellLine_word2vec, combined_2])
        combined = layers.Dense(1024, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.Dropout(0.5)(combined)
        pred = layers.Dense(1, activation='sigmoid')(combined)

        model = keras.Model(
            inputs = [input_cellLine_word2vec, 
                      input_drug_word2vec_1, input_drug_word2vec_2],
            outputs = pred
        )

        return model
    
    
    def autoencoder_only_word2vec():
        input_cellLine_word2vec = keras.Input(shape=(256,), name='cellLine_word2vec')
        
        # cellLine_word2vec auto-encoder
        cellLine_word2vec = layers.Dense(2048, activation='relu')(input_cellLine_word2vec)
        cellLine_word2vec = layers.Dense(1024, activation='relu')(cellLine_word2vec)
        cellLine_word2vec = layers.Dense(512, activation='relu')(cellLine_word2vec)
        cellLine_word2vec_encoder_output = layers.Dense(256, activation='relu')(cellLine_word2vec)

        cellLine_word2vec_encoder = keras.Model(input_cellLine_word2vec, cellLine_word2vec_encoder_output)

        cellLine_word2vec = layers.Dense(512, activation='relu')(cellLine_word2vec_encoder_output)
        cellLine_word2vec = layers.Dense(1024, activation='relu')(cellLine_word2vec)
        cellLine_word2vec = layers.Dense(2048, activation='relu')(cellLine_word2vec)
        cellLine_word2vec_decoder_output = layers.Dense(256, activation='relu')(cellLine_word2vec)

        cellLine_word2vec_autoencoder = keras.Model(input_cellLine_word2vec, cellLine_word2vec_decoder_output)
        
        
        autoencoder = keras.Model(
            inputs = [input_cellLine_word2vec],
            outputs = [cellLine_word2vec_decoder_output]
        )
        #keras.utils.plot_model(autoencoder, "autoencoder.png", show_shapes=True)

        return autoencoder, (cellLine_word2vec_encoder)
    
    
    

    model = locals()[model_name]()
    return model