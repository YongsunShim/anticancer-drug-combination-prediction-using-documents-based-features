import csv
import time

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

has_stopwords = False
lemmatization = True
    
def get_token(text):
    tk = WhitespaceTokenizer()
    n = WordNetLemmatizer()
    
    tokens = []
    if has_stopwords == False:
        stop_words = set(stopwords.words('english'))
        tokens = [n.lemmatize(word) for word in tk.tokenize(text.lower()) if word not in stop_words]
    else:
        tokens = [n.lemmatize(w) for w in tk.tokenize(text.lower())]
            
    return tokens

def main():
    documents_path = './data/documents.txt'
    revised_documents_path = './data/revised_documents.txt'

    documents_f = open(documents_path, 'r')
    revised_documents_f = open(revised_documents_path, 'w')

    documents_lines = documents_f.readlines()

    for i in range(0, len(documents_lines)):
        split = documents_lines[i][:-1].split('\t')
        if(len(split) > 0):
            revised_text = ''
            for j in range(0, len(get_token(split[2]))):
                revised_text = revised_text + ' ' + get_token(split[2])[j]
            revised_documents_f.write(revised_text + '\n')
        i=i+1
    documents_f.close()
    revised_documents_f.close()
    
if __name__ == "__main__":
    main()
