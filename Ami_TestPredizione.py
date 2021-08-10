# -*- coding: utf-8 -*-
"""

@author: ALEX
"""
import tensorflow as tf

import pandas as pd
import json
import time
from keras import layers

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
import string
import re
import nltk


import os

#!pip install h5py
#!pip install emot
#!pip install emoji

import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS

import emoji


df_ami= pd.read_csv('AMI2020_training_raw.tsv', sep='\t')


#nltk.download('stopwords')

json_file = open('Model/model_misogino.json', 'r')
modello_misogino_json = json_file.read()
json_file.close()
modello_misogino = tf.keras.models.model_from_json(modello_misogino_json)
# load weights into new model
modello_misogino.load_weights("Model_h5/model_misogino_h5.h5")
print("Modello per il riconoscimento di contenuti misogini caricato")

json_file2 = open('Model/model2_aggressività.json', 'r')
modello_aggressivo_json = json_file2.read()
json_file2.close()
modello_aggressivo = tf.keras.models.model_from_json(modello_aggressivo_json)
# load weights into new model
modello_aggressivo.load_weights("Model_h5/model2_h5_aggressività.h5")
print("Modello per il riconoscimento di contenuti aggressivi caricato")

def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
   # for emot in EMOTICONS:
    #    text = re.sub(u'( '+ emot +')', " "+"".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

df_ami['text_pulito'] = df_ami['text'].map(lambda x: convert_emojis(x))

def clean_text(text):  
      ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    ## Remove stop words
    stops = set(stopwords.words("italian"))
    text = [w for w in text if not w in stops and not "http" in w and not "@" in w] 
    text = " ".join(text)    ## Clean the text
    #text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r",", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", "  ", text)
    text = re.sub(r"\+", "  ", text)
    text = re.sub(r"\-", "  ", text)
    text = re.sub(r"\=", "  ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\:", "  ", text)
    text = re.sub(r"“", "  ", text)
    text = re.sub(r"”", "  ", text)
    text = re.sub(r"\’", "  ", text)
    text = re.sub(r"-", "  ", text)
    text = re.sub(r"\.", "  ", text)
    text = re.sub(r"\(", "  ", text)
    text = re.sub(r"\)", "  ", text)
    text = re.sub(r"«", "  ", text)
    text = re.sub(r"»", "  ", text)
    text = re.sub(r"\€", " euro ", text)
    text = re.sub(r"\*", "  ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", "  ", text)
    text = re.sub(r"\0s", "0", text)
    
    #specifici per il dataset
    text = re.sub(r" x ", " per ", text)  
    text = re.sub(r" nn ", " non ", text)
    text = re.sub(r" co ", " con ", text)
    text = re.sub(r"•", " ", text)
    text = re.sub(r"—", " ", text)
    text = re.sub('[\W_]+', ' ', text) #♥️
    #text = re.sub('♥️', ' ', text)
    #text = re.sub('❤', ' ', text)
    #text = re.sub('✨', ' ', text)
    #text = re.sub('►', ' ', text)
    text = re.sub('&lt;3', ' ', text)
    text = re.sub(' rt ', ' ', text)
    
    text = text.split()
    text = [w for w in text if not w in stops]
    
    text = " ".join(text)    ## Clean the text

    ## Stemming 
    #text = text.split()
    #stemmer = SnowballStemmer('italian')
    #stemmed_words = [stemmer.stem(word) for word in text] 
    #text = " ".join(stemmed_words)
    return text

df_ami['text_pulito'] = df_ami['text_pulito'].map(lambda x: clean_text(x))

#print(df_ami.text_pulito[1])

#delete empty row
df_ami = df_ami[df_ami['text'] != '']

def conta_punti_esclamativi(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if "!" in w]
    count= len(text)
  
    return count

def conta_punti_interrogativi(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if "?" in w]
    count= len(text)
  
    return count

def conta_tag_utenti(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if "@" in w]
    count= len(text)
  
    return count

def conta_hashtag(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if "#" in w]
    count= len(text)
  
    return count

def conta_nomi_femminili(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    nomi= ['zie', 'figlie', 'ragazze', 'madri', 'mogli', 'fidanzata', 'signore', 'signorine', 'spose', 'sorelle', 'zia', 'figlia', 'ragazza', 'madre', 'lei', 'moglie', 'signora', 'signorina', 'sposa', 'sorella']
    text = [w for w in text if w in nomi]
    count= len(text)
  
    return count

def conta_risate(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if "ahah" in w or "haha" in w]
    count= len(text)
  
    return count

def conta_insulti(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    nomi= ['puttana', 'vacca', 'troia', 'pompinara', 'frigida', 'figa', 'zitella', 'strega', 'arpia', 'isterica', 'oca', 'schifosa',
           'gallina', 'gattamorta', 'fighetta', 'sciaquetta', 'racchia', 'cessa', 'cozza', 'culona', 'travestito','rifatta','zoccola']
    text = [w for w in text if w in nomi]
    count= len(text)
  
    return count


most_occur_parole = Counter(" ".join(df_ami["text_pulito"].str.lower()).split()).most_common(20)
df_parole = pd.DataFrame(most_occur_parole, columns = ['Parole', 'N° parole'])

df_m0 = df_ami[df_ami.misogynous==0]
df_m1 = df_ami[df_ami.misogynous==1]
most_occur0_parole = Counter(" ".join(df_m0["text_pulito"].str.lower()).split()).most_common(20)
most_occur1_parole = Counter(" ".join(df_m1["text_pulito"].str.lower()).split()).most_common(20)
df0_parole = pd.DataFrame(most_occur0_parole, columns = ['Parole', 'N° parole'])
df1_parole = pd.DataFrame(most_occur1_parole, columns = ['Parole', 'N° parole'])


def conta_parole(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text]
    count= len(text)
  
    return count

def conta_caratteri(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    count= len(text)
  
    return count

def conta_20totaliusate(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if w in df_parole['Parole'].values]
    count= len(text)
  
    return count
def conta_20usatenonmiso(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if w in df0_parole['Parole'].values]
    count= len(text)
  
    return count
def conta_20usatemiso(text):  
    ## Convert words to lower case and split them
    text = text.lower().split()
    text = [w for w in text if w in df1_parole['Parole'].values]
    count= len(text)
    return count
def conta_5totaliemoji(text):  
    ## Convert words to lower case and split them
    #text = text.lower().split()
    lista= ['❤','♥','⚫','❤❤❤','✔']
    text = [w for w in text if w in lista]
    count= len(text)
  
    return count
def conta_5emojinonmiso(text):  
    ## Convert words to lower case and split them
    #text = text.lower().split()
    lista= ['❤','♥','⚫','♀','☺']
    text = [w for w in text if w in lista]
    count= len(text)
  
    return count
def conta_5emojimiso(text):  
    ## Convert words to lower case and split them
    #text = text.lower().split()
    lista= ['❤','♥','♥♥','❤❤❤','✋']
    text = [w for w in text if w in lista]
    count= len(text)
    return count


#calcolo le features
df_ami['num_parole'] = df_ami['text'].map(lambda x: conta_parole(x))
df_ami['num_caratteri'] = df_ami['text'].str.len()
df_ami['num_esclamativi'] = df_ami['text'].map(lambda x: conta_punti_esclamativi(x))
df_ami['num_interrogativi'] = df_ami['text'].map(lambda x: conta_punti_interrogativi(x))
df_ami['num_tag_utenti'] = df_ami['text'].map(lambda x: conta_tag_utenti(x))
df_ami['num_hashtag'] = df_ami['text'].map(lambda x: conta_hashtag(x))
df_ami['num_nomi_femminili'] = df_ami['text'].map(lambda x: conta_nomi_femminili(x))
df_ami['num_risate'] = df_ami['text'].map(lambda x: conta_risate(x))
df_ami['num_insulti'] = df_ami['text'].map(lambda x: conta_insulti(x))
df_ami['num_20totaliusate'] = df_ami['text'].map(lambda x: conta_20totaliusate(x))
df_ami['num_20usatemiso'] = df_ami['text'].map(lambda x: conta_20usatemiso(x))
df_ami['num_20usatenonmiso'] = df_ami['text'].map(lambda x: conta_20usatenonmiso(x))
df_ami['num_5totaliemoji'] = df_ami['text'].map(lambda x: conta_5totaliemoji(x))
df_ami['num_5emojimiso'] = df_ami['text'].map(lambda x: conta_5emojimiso(x))
df_ami['num_5emojinonmiso'] = df_ami['text'].map(lambda x: conta_5emojinonmiso(x))
#print(df_ami)

'''
fase di embedding
'''


lista_testo = df_ami["text_pulito"].fillna('').to_list()
lista_testo = [str(i) for i in lista_testo]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lista_testo)
vocab_size = len(tokenizer.word_index) + 1
lista_testo_tokenizer = tokenizer.texts_to_sequences(lista_testo)
maxlen = max( len(x) for x in lista_testo_tokenizer)

from io import StringIO

value = input("Inserire la frase da classificare:\n")
value=StringIO(value)
df = pd.DataFrame(value) 
df['frase_pulita'] = df[0].map(lambda x: convert_emojis(x))
df['frase_pulita'] = df['frase_pulita'].map(lambda x: clean_text(x))
df = df[df['frase_pulita'] != '']
#calcolo le features
df['num_parole'] = df[0].map(lambda x: conta_parole(x))
df['num_caratteri'] = df[0].str.len()
df['num_esclamativi'] = df[0].map(lambda x: conta_punti_esclamativi(x))
df['num_interrogativi'] = df[0].map(lambda x: conta_punti_interrogativi(x))
df['num_tag_utenti'] = df[0].map(lambda x: conta_tag_utenti(x))
df['num_hashtag'] = df[0].map(lambda x: conta_hashtag(x))
df['num_nomi_femminili'] = df[0].map(lambda x: conta_nomi_femminili(x))
df['num_risate'] = df[0].map(lambda x: conta_risate(x))
df['num_insulti'] = df[0].map(lambda x: conta_insulti(x))
df['num_20totaliusate'] =df[0].map(lambda x: conta_20totaliusate(x))
df['num_20usatemiso'] = df[0].map(lambda x: conta_20usatemiso(x))
df['num_20usatenonmiso'] = df[0].map(lambda x: conta_20usatenonmiso(x))
df['num_5totaliemoji'] = df[0].map(lambda x: conta_5totaliemoji(x))
df['num_5emojimiso'] = df[0].map(lambda x: conta_5emojimiso(x))
df['num_5emojinonmiso'] = df[0].map(lambda x: conta_5emojinonmiso(x))
#print(df_ami)





df['testo_token']= tokenizer.texts_to_sequences(df['frase_pulita'])
#print("La lunghezza prima il post padding è: ", len(df['testo_token'].iloc[0]))
df['testo_token_padding'] = pad_sequences(df['testo_token'], padding = "post", maxlen = maxlen).tolist()
#print("La lunghezza dopo il post padding è: ", len(df['testo_token_padding'].iloc[0]))


df_test = df[['testo_token_padding','num_parole','num_caratteri','num_esclamativi',
                      'num_interrogativi','num_tag_utenti','num_hashtag','num_nomi_femminili','num_risate','num_insulti',
                      'num_20totaliusate','num_20usatemiso','num_20usatenonmiso','num_5totaliemoji','num_5emojimiso',
                      'num_5emojinonmiso']]

values_df_test= df_test.values

valori_embedding = np.array([item[0]for item in values_df_test]).astype(np.float32)
valori_features = np.array([item[1:]for item in values_df_test]).astype(np.float32)


predizione_m = modello_misogino.predict([valori_embedding,valori_features])
predizione_non_misogino= predizione_m[0][0]*100
predizione_misogino= predizione_m[0][1]*100
predizione_finale_misogino = np.argmax(predizione_m, axis=1)

predizione_a = modello_aggressivo.predict([valori_embedding,valori_features])
predizione_non_aggressivo= predizione_a[0][0]*100
predizione_aggressivo= predizione_a[0][1]*100
predizione_finale_aggressivo = np.argmax(predizione_a, axis=1)

if predizione_finale_misogino[0] == 0:
    #if predizione_final2[0] == 0:
        print("Il testo fornito non è misogino al ", "%.2f" % predizione_non_misogino,"%")

else:
    if predizione_finale_aggressivo[0] == 0:
        print("Il testo fornito è misogino al ", "%.2f" % predizione_misogino,
              "% e non è aggressivo al ","%.2f" % predizione_non_aggressivo,"%")
    else:
        print("Il testo fornito è misogino al ", "%.2f" % predizione_misogino,
              "% ed è aggressivo al ","%.2f" % predizione_aggressivo,"%")







