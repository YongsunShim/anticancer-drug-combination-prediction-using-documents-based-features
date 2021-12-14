# A novel approach to predict anti-cancer drug combinations using document-based feature extraction

## Description
We provide the code and data to implement our approach. Our code is divided into Java and Python. In Java code, biomedical documents are collected from PubMed, and synonyms in the collected biomedical documents are converted into representative terms. In Python code, document preprocessing such as tokenization and lemmatization is performed, and document-based features are extracted. Finally, training data is constructed by combining features, and a prediction model is created using deep learning and machine learning algorithms.

## Data download
You should download it [here](https://drive.google.com/file/d/1hBGQCiRV6eqENgYHGH-yCVStPchNBRJ-/view?usp=sharing). You need to download the dataset and unzip it to the same path with the code.

## Requirements
**1. Java**
+ maven

**2. Python**
+ Numpy
+ Pandas
+ tqdm
+ bisect
+ scikit-learn
+ tensorflow
+ keras
+ nltk
+ gensim

## Steps
**Step 1: decompress data files**
+ unzip data.zip

**Step 2: collect biomedical documents**
```c
sh collect_documents.sh
```

**Step 3: revise biomedical documents**
```c
python document_preprocessing.py
```

**Step 4: extract document-based features**
```c
python extract_features_based_documents.py
```

**Step 5: execute main class**
```c
python main.py 'model_name'
```
+ You need to add 'model_name' to run the code. The list of 'model_name' is in method_config.py.
