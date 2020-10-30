#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:23:50 2020

@author: guido
"""

import os, torch, utils, tqdm, csv
import pandas as pd
#from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional

import embeddings
#from textblob import TextBlob
import time
from textblob import TextBlob
#from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('bert-base-nli-max-tokens')
sentences_1 = ['Sentences are passed as a list of string.','This framework generates embeddings for each input sentence. The great conqueror was augusto.',
     
    'The quick brown fox jumps over the lazy dog.']

#print(sentence_embeddings)

start_time = time.time()
PATH = os.getcwd()




###carico lo squad 
squad = utils.squad_preprocessing(PATH)
context_embeddings =[]
list(squad.context.unique()[:3])

passages = utils.wrong_passages(squad.context.unique())
for el in sentences_1:
    tb = TextBlob(el)
    sentences = tb.sentences
    if len(sentences)>1:
        lista=[]
        for sent in sentences:
            lista.append(str(sent))
    else:
        lista = str(sentences[0])
    sentences_embeddings = embeddings.bert_embeddings(lista, sent_emb=True)
    context_embeddings.append(sentences_embeddings)
    
    print (el)
#context_embeddings = embeddings.bert_embeddings(sentences, sent_emb=True)
#answer_embeddings = glove_embeddings(squad.answer_text, model="840B.300d")
#question_embeddings = glove_embeddings(squad.question, model = "840B.300d") 
#prova = embeddings.bert_embeddings("Madonna's grandfathers were from Italy")

        

#for sent in squad.context[:10]:
#    x = TextBlob(sent)
#    print(x.sentences)
    



#emb = embeddings.bert_embeddings_old(lista[:500])
print("--- %s seconds ---" % (time.time() - start_time))
#start_time = time.time()
#emb_new = embeddings.bert_embeddings(lista[:500])

print("--- %s seconds ---" % (time.time() - start_time))

"""
context_input = Input(shape=(700, ), dtype='int32', name='context_input')
x = Embedding(input_dim=vocab_size, output_dim=50, weights=[embedding_matrix], 
              input_length=700, trainable=False)(context_input)
lstm_out = Bidirectional(LSTM(256, return_sequences=True, implementation=2), merge_mode='concat')(x)
drop_1 = Dropout(0.5)(lstm_out)

#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense# Neural network
model = Sequential()
model.add(Dense(16, input_dim=20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))
"""