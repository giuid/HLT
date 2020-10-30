#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:10:19 2020

@author: guido
"""

import pandas as pd
import os, torch, utils, tqdm, csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
#from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from transformers import BertModel, AutoTokenizer
PATH = os.getcwd()




def load_glove(path):
    try:
        print()
    except:
        embedding_vector = {}
        f = open(PATH + '/glove/glove.840B.300d.txt')
        for line in tqdm.tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef
        w = csv.writer(open(PATH + "/glove/embedding_vector.csv", "w"))
        for key, val in embedding_vector.items():
            w.writerow([key, val])  
            
def glove_dictionary(glove_ver = "6B.100d" ):
    assert glove_ver in ["6B.100d","6B.50d","6B.200d","6B.300d","840B.300d"],"The glove pretrained embedding you chose is not available"
    GLOVE_DIR = PATH + "/glove/"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, "glove."+glove_ver+".txt"))
    for line in f:
        values = line.split()
        if (len(values) > 301):
            values[:(len(values)-300)] = ["".join(values[:(len(values)-300)])] 
        word = values[0]
       # print(len(values))
        vectors = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vectors
    f.close()
    return embeddings_index

glove_vectors = None

def glove_embeddings(lista, max_length = 300 ,EMBEDDING_DIM = 300, model = "6B.300d"):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lista)
    seq_lista = tokenizer.texts_to_sequences(lista)
    lista_encoded = pad_sequences(seq_lista, maxlen = max_length, padding = "post")
    global glove_vectors
    if glove_vectors == None:
        glove_vectors = glove_dictionary(model)
    vocab_size = len(tokenizer.word_index)+1
    word_vector_matrix = np.zeros((vocab_size, 300))
    i=0
    for word, index in tokenizer.word_index.items():
        vector = glove_vectors.get(word)
        if vector is not None:
            word_vector_matrix[index] = vector
        else:
            print(i)
            print(str(len(lista) - len(word_vector_matrix))) 
            i+=1
    return word_vector_matrix




model_bert = None
def bert_embeddings(lista, padding_length = 300,pretrained_model = "bert-base-uncased", sent_emb = False):
     #lista = "[CLS] " + lista + " [SEP]"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    embeddings = []
    if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
    global model_bert
    if (model_bert == None):
        model_bert = BertModel.from_pretrained(pretrained_model,output_hidden_states = True)
        model_bert.to(device)

    if type(lista)==str:    
        tokenized_text_auto = tokenizer(lista, padding='max_length', truncation=True, max_length=padding_length)
        segments_ids = tokenized_text_auto["attention_mask"]
        indexed_tokens = tokenized_text_auto["input_ids"]
        
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_tensors = torch.tensor([segments_ids]).cuda()
        with torch.no_grad():
            
            outputs = model_bert(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
            #print(outputs.is_cuda)
        hidden_states
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(hidden_states, dim=0)    
        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        # Stores the token vectors, with shape [22 x 3,072]
        torch.cuda.empty_cache()            
        token_embeddings = token_embeddings
        #print(token_embeddings.is_cuda)
        token_vecs_cat = []
        # `token_embeddings` is a [22 x 12 x 768] tensor.
        
        # For each token in the sentence...
        for token in token_embeddings:
            
            # `token` is a [12 x 768] tensor
        
            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            #cat_vec = cat_vec.cpu()
            
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)
            
        token_vecs_cat = torch.stack(token_vecs_cat)
        
        if sent_emb:
            token_vecs_cat = torch.mean(token_vecs_cat, dim=0)
        token_vecs_cat = token_vecs_cat.cpu()
        embeddings.append(token_vecs_cat)
        torch.cuda.memory_summary(device=None, abbreviated=False)
    
    else:                                      
        model_bert.eval()
        for i in range(0,len(lista),100):
            listina = lista[i:i+100]
            #torch.cuda.empty_cache()
            for text in listina:
                tokenized_text_auto = tokenizer(text, padding='max_length', truncation=True, max_length=padding_length)
                segments_ids = tokenized_text_auto["attention_mask"]
                indexed_tokens = tokenized_text_auto["input_ids"]
                
                tokens_tensor = torch.tensor([indexed_tokens]).cuda()
                segments_tensors = torch.tensor([segments_ids]).cuda()
                with torch.no_grad():
                    
                    outputs = model_bert(tokens_tensor, segments_tensors)
                    hidden_states = outputs[2]
                    #print(outputs.is_cuda)
                hidden_states
                # Concatenate the tensors for all layers. We use `stack` here to
                # create a new dimension in the tensor.
                token_embeddings = torch.stack(hidden_states, dim=0)    
                # Remove dimension 1, the "batches".
                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings = token_embeddings.permute(1,0,2)
                # Stores the token vectors, with shape [22 x 3,072]
                torch.cuda.empty_cache()            
                token_embeddings = token_embeddings
                #print(token_embeddings.is_cuda)
                token_vecs_cat = []
                # `token_embeddings` is a [22 x 12 x 768] tensor.
                
                # For each token in the sentence...
                for token in token_embeddings:
                    
                    # `token` is a [12 x 768] tensor
                
                    # Concatenate the vectors (that is, append them together) from the last 
                    # four layers.
                    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
                    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                    #cat_vec = cat_vec.cpu()
                    
                    # Use `cat_vec` to represent `token`.
                    token_vecs_cat.append(cat_vec)
                    
                token_vecs_cat = torch.stack(token_vecs_cat)
                
                if sent_emb:
                    token_vecs_cat = torch.mean(token_vecs_cat, dim=0)
                token_vecs_cat = token_vecs_cat.cpu()
                embeddings.append(token_vecs_cat)
                torch.cuda.memory_summary(device=None, abbreviated=False)


    return embeddings

    


def bert_embeddings_old(lista, padding_length = 300):
    #lista = "[CLS] " + lista + " [SEP]"
    
    pretrained_model = "bert-base-uncased"
    if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    lista = list(lista)
    global model_bert
    if (model_bert == None):
        model_bert = BertModel.from_pretrained(pretrained_model,output_hidden_states = True)
        model_bert.to(device)
                                      
    model_bert.eval()
    embeddings = []
    for i in range(0,len(lista),100):
        listina = lista[i:i+100]
        #torch.cuda.empty_cache()
        for text in listina:
            tokenized_text_auto = tokenizer(text, padding='max_length', truncation=True, max_length=padding_length)
            segments_ids = tokenized_text_auto["attention_mask"]
            indexed_tokens = tokenized_text_auto["input_ids"]
            
            tokens_tensor = torch.tensor([indexed_tokens]).cuda()
            segments_tensors = torch.tensor([segments_ids]).cuda()
            with torch.no_grad():
                
                outputs = model_bert(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
                #print(outputs.is_cuda)
            hidden_states
            # Concatenate the tensors for all layers. We use `stack` here to
            # create a new dimension in the tensor.
            token_embeddings = torch.stack(hidden_states, dim=0)    
            # Remove dimension 1, the "batches".
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)
            # Stores the token vectors, with shape [22 x 3,072]
            torch.cuda.empty_cache()            
            token_embeddings = token_embeddings
            #print(token_embeddings.is_cuda)
            token_vecs_cat = []
            
            # `token_embeddings` is a [22 x 12 x 768] tensor.
            
            # For each token in the sentence...
            for token in token_embeddings:
                
                # `token` is a [12 x 768] tensor
            
                # Concatenate the vectors (that is, append them together) from the last 
                # four layers.
                # Each layer vector is 768 values, so `cat_vec` is length 3,072.
                cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                cat_vec = cat_vec.cpu()
                
                # Use `cat_vec` to represent `token`.
                token_vecs_cat.append(cat_vec)
            embeddings.append(token_vecs_cat)
            torch.cuda.memory_summary(device=None, abbreviated=False)


    return embeddings



