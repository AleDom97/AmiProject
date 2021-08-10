# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:36:04 2020

@author: ALEX
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:42:25 2020

@author: ALEX
"""

import pandas as pd
import json
import numpy as np


#import time
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


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

!pip install h5py
!pip install emot
!pip install emoji

from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji

!pip install mlxtend
!pip install mlxtend --upgrade --no-deps
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import mlxtend   




df_ami= pd.read_csv('AMI2020_training_raw.tsv', sep='\t')
print(df_ami)
print(df_ami.aggressiveness.value_counts())


nltk.download('stopwords')

# Separate majority and minority classes
df_0 = df_ami[df_ami.aggressiveness==0]
df_1 = df_ami[df_ami.aggressiveness==1]

# Upsample
df_1_upsample = resample(df_1, 
                                 replace=True,     # sample with replacement
                                 n_samples=2200) # reproducible results

# Combine majority class with upsampled minority class
df_sampled = pd.concat([df_0, df_1_upsample])

df_sampled.aggressiveness.value_counts()
df_sampled.reset_index(inplace=True)
df_ami=df_sampled
df_ami= df_ami.drop(df_ami.columns[0], axis=1)

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

print(df_ami.text_pulito[1])

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



#calcolo le parole più usate nel dataset
most_occur_parole = Counter(" ".join(df_ami["text_pulito"].str.lower()).split()).most_common(20)
df_parole = pd.DataFrame(most_occur_parole, columns = ['Parole', 'N° parole'])
#df.plot.bar(x='Word',y='Count')




#calcolo le parole più usate nel dataset misogino e non misogino
df_m0 = df_ami[df_ami.aggressiveness==0]
df_m1 = df_ami[df_ami.aggressiveness==1]
most_occur0_parole = Counter(" ".join(df_m0["text_pulito"].str.lower()).split()).most_common(20)
most_occur1_parole = Counter(" ".join(df_m1["text_pulito"].str.lower()).split()).most_common(20)
df0_parole = pd.DataFrame(most_occur0_parole, columns = ['Parole', 'N° parole'])
df1_parole = pd.DataFrame(most_occur1_parole, columns = ['Parole', 'N° parole'])


#intersezione parole usate
s1 = pd.merge(df0_parole, df1_parole, how='inner', on=['Parole'])
s1
'''
plt.plot(s1['Parole'],s1['N° parole_x'])
plt.plot(s1['Parole'],s1['N° parole_y'])
plt.title('Intersezione tra le due liste')
plt.ylabel('N° parole')
plt.legend(['Testo non misogino', 'Testo misogino'], loc='upper right')
plt.show()
'''

def extract_emojis(s):
  return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)

df_ami['text_emoji'] = df_ami['text'].map(lambda x: extract_emojis(x))
print(df_ami.text_emoji.unique())

#calcolo le emoji più usate nel dataset
most_occur_emoji = Counter(" ".join(df_ami["text_emoji"].str.lower()).split()).most_common(10)
df_emoji = pd.DataFrame(most_occur_emoji, columns = ['Emoji', 'N° emoji'])
#df.plot.bar(x='Word',y='Count')


#calcolo le emoji più usate nel dataset misogino e non misogino
df_m0 = df_ami[df_ami.aggressiveness==0]
df_m1 = df_ami[df_ami.aggressiveness==1]
most_occur0_emoji = Counter(" ".join(df_m0["text_emoji"].str.lower()).split()).most_common(10)
most_occur1_emoji = Counter(" ".join(df_m1["text_emoji"].str.lower()).split()).most_common(10)
df0_emoji = pd.DataFrame(most_occur0_emoji, columns = ['Emoji', 'N° emoji'])
df1_emoji = pd.DataFrame(most_occur1_emoji, columns = ['Emoji', 'N° emoji'])


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
    lista= ['❤','♥','♀','❤❤❤','♥♥']
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
    lista= ['❤','♥','♥♥','❤❤❤','♀']
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
print(df_ami)

df_ami.to_csv('df_ami_Aggressività.csv', encoding='utf-8')


'''
fase di embedding
'''

lista_testo = df_ami["text_pulito"].fillna('').to_list()
lista_testo = [str(i) for i in lista_testo]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lista_testo)

vocab_size = len(tokenizer.word_index) + 1
#vocab_size =  vocab_size + 658 # è necessario avere la stessa dimensione dei due vocabolari
lista_testo_tokenizer = tokenizer.texts_to_sequences(lista_testo)
maxlen = max( len(x) for x in lista_testo_tokenizer)

df_ami['testo_token']= tokenizer.texts_to_sequences(df_ami['text_pulito'])
print("La lunghezza prima il post padding è: ", len(df_ami['testo_token'].iloc[0]))
df_ami['testo_token_padding'] = pad_sequences(df_ami['testo_token'], padding = "post", maxlen = maxlen).tolist()
print("La lunghezza dopo il post padding è: ", len(df_ami['testo_token_padding'].iloc[1]))


df_training = df_ami[['testo_token_padding','num_parole','num_caratteri','num_esclamativi',
                      'num_interrogativi','num_tag_utenti','num_hashtag','num_nomi_femminili','num_risate','num_insulti',
                      'num_20totaliusate','num_20usatemiso','num_20usatenonmiso','num_5totaliemoji','num_5emojimiso',
                      'num_5emojinonmiso','aggressiveness']]

X = df_training.iloc[:,0:16].values
Y = df_training.iloc[:,16].values


RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)

X_train_embedding= np.array([item[0]for item in X_train])
X_train_features= np.array([item[1:]for item in X_train])

X_test_embedding= np.array([item[0]for item in X_test])
X_test_features= np.array([item[1:]for item in X_test])


y_train = to_categorical(y_train,2)
y_test_cat= to_categorical(y_test,2)

'''
definisco il modello sequanziale
'''
'''
# Define MLP architecture quello migliore ma che prende un solo input
embedding_dim = 100

model_dense = tf.keras.Sequential()

model_dense.add(tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length= maxlen)) #3d (batch_size, maxlen,dim)

model_dense.add(tf.keras.layers.Flatten())

model_dense.add(tf.keras.layers.Dense(128, activation="relu"))
model_dense.add(tf.keras.layers.Dense(64, activation="relu"))

#model_dense.add(tf.keras.layers.Dense(2, activation="softmax"))
model_dense.add(tf.keras.layers.Dense(1, activation="sigmoid"))

#fine arcchiettura del modello

model_dense.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_dense.summary()


#history = model_dense.fit(np.array(X_train), np.array(y_train), epochs=5,verbose=True, validation_data=(np.array(X_test), np.array(y_test)),batch_size=32)
history = model_dense.fit(X_train_embedding, y_train, epochs=5,verbose=True, validation_data=(X_test_embedding, y_test),batch_size=32)

loss, accuracy = model_dense.evaluate(X_test_embedding, y_test, verbose= 1)
'''

'''
definisco il modello con doppio input
'''

embedding_dim = 100

input_testo = keras.layers.Input(shape=(maxlen,))

x= keras.layers.Embedding(vocab_size,embedding_dim, input_length= maxlen, trainable= True)(input_testo)
flatten = keras.layers.Flatten()(x)
input_features = keras.layers.Input(shape=(X_train_features.shape[1],))

model_final= keras.layers.Concatenate()([flatten, input_features])
model_final= keras.layers.Dense(128,activation='relu',bias_initializer='zeros')(model_final)
model_final= keras.layers.Dense(64,activation='relu',bias_initializer='zeros')(model_final)
#model_final= keras.layers.Dense(2,activation='softmax',bias_initializer='zeros')(model_final)
model_final= keras.layers.Dense(2,activation='sigmoid',bias_initializer='zeros')(model_final)
model_final= keras.Model([input_testo, input_features],model_final)
model_final.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])

model_final.summary()

#history = model_final.fit(x=[X_train_embedding,X_train_features], y= np.array(y_train), batch_size = 32, epochs = 5, verbose = 1, validation_split= 0.2 )
history = model_final.fit(x=[X_train_embedding,X_train_features], y= np.array(y_train), epochs=5,verbose=True, validation_data=([X_test_embedding,X_test_features], np.array(y_test_cat)),batch_size=32)

#loss, accuracy = model_final.evaluate(x=[X_test_embedding,X_test_features], y_test_cat, verbose= 1)
#score = model_final.evaluate(x=[X_test_embedding,X_test_features],y= np.array(y_test_cat))
      

predizione_a = model_final.predict([X_test_embedding,X_test_features])
predizione_final2 = np.argmax(predizione_a, axis=1)
rounded_labels=np.argmax(y_test_cat, axis=1)
mat = confusion_matrix(rounded_labels, predizione_final2)
plot_confusion_matrix(conf_mat=mat, figsize=(12, 12), show_normed=True)

# serialize model to JSON
model_json = model_final.to_json()
with open("Model/model2_aggressività.json", "w+") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_final.save_weights("Model_h5/model2_h5_aggressività.h5")
print("Modello aggressività salvato correttamente")
