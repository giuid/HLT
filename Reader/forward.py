#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 12:06:57 2020

@author: guido & giuseppe
"""
import long_answers
import torch
import os
import short_answers
import re
from transformers import RobertaModel
import sys
from operator import itemgetter
from tqdm import tqdm

""" Blocca i print """
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

""" Ripristina i print """
def enablePrint():
    sys.stdout = sys.__stdout__
    
""" Date le predizioni individua le due migliori """    
def get_best_results(predizioni):
    first = 0
    second = 0
    
    for i in range(len(predizioni)):
        if predizioni[i]>predizioni[first]:
            second = first
            first = i
        elif predizioni[i]>predizioni[second] and predizioni[i]<predizioni[first]:
            second = i
    return [first,second]

""" Valuta il modello """       
def evaluate(model, test_set, batch_size= 24):
    from transformers import RobertaModel
    import torch.nn as nn
    criterion = nn.BCELoss() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = RobertaModel.from_pretrained("roberta-base")
    input_ids = test_set.input_ids.values.tolist()
    attention_mask = test_set.attention_mask.values.tolist()
    labels = test_set.labels.values.tolist()
    model= model.eval()
    model= model.to(device)
    losses=[]
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in tqdm(range(0,len(test_set),batch_size)):
        with torch.no_grad():
            emb = long_answers.sentence_bert(bert_model, input_ids[i:i+batch_size], attention_mask[i:i+batch_size])
            emb = emb.to(device)
            lab = torch.tensor(labels[i:i+batch_size]).to(device).unsqueeze(1)
            output = model(emb)
            loss = criterion(output, lab.float())
            losses.append(loss)
            predizioni = [round(x[0]) for x in output.tolist()]
            for i in range(len(predizioni)):
                if predizioni[i] == lab[i] == 1:
                    true_pos+=1
                elif predizioni[i] == lab[i] == 0:
                    true_neg+=1
                elif predizioni[i]==0:
                    false_neg+=1
                else:
                    false_pos+=1

    return {"losses" : losses,"false_pos" : false_pos, "false_neg" :false_neg, "true_pos" : true_pos, "true_neg" :  true_neg}
  

""" Data la domanda e le pagine di WIkipedia contenute in una lista fornisce in output le risposte, corte e lunghe, ordinate in base alla loro probabilità  """  
def get_answer(question,wikipages, batch_size = 24):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    long= long_answers.LongAnswer()
    
    model = long_answers.LSTM().to(device)
    bert_model = RobertaModel.from_pretrained("roberta-base")
    
    paragraphs=[]
    for el in wikipages:
        paragraphs +=long.html_tag(el)
    questions = len(paragraphs)*[question]
    long.create_features(questions, paragraphs,forward=True)
    data = long.forward_data
    #directory = os.getcwd()
    
    model=model.to(device)
    predizioni =[]
    for i in range(0, len(data),batch_size):
        embs= long.sentence_bert(bert_model, data.input_ids.values.tolist()[i:i+batch_size], data.attention_mask.values.tolist()[i:i+batch_size]).to(device)
        prediction = model.forward(embs)
        pred = [round(x[0]) for x in prediction.tolist()]
        logits = [(x[0]) for x in prediction.tolist()]
        predizioni += pred
        del embs,prediction, pred, logits
    bests_indices = get_best_results(predizioni)
    del model, bert_model, paragraphs, long,
    bests_indices = [i for i, x in enumerate(predizioni) if x == 1]
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    short_answer = short_answers.ShortAnswers()
    short_answer.create_model("roberta")
    short_answer.load_model("/mnt/Volume/HLT/short_answer_data/models/final/pytorch_model.bin")
    
    model_short = short_answer.model
    
    tokenizer = short_answer.tokenizer

    answers = []
    score = [] 

    for index in bests_indices:
        scores = model_short.forward( torch.tensor([data.input_ids.values.tolist()[index]]), torch.tensor([data.attention_mask.values.tolist()[index]]))
        score.append(scores)
        start_index = torch.argmax(scores[0])
        end_index = torch.argmax(scores[1])
        start_score = float(scores[0][0][start_index])
        end_score = float(scores[1][0][end_index])
        mean_score = float((start_score+end_score)/2)
        par = tokenizer.convert_ids_to_tokens(data.input_ids.values.tolist()[index])
        par = par[:par.index("<pad>")]
        answer = re.sub("Ġ"," ",''.join( par[int(start_index)+1:int(end_index)+2]))
        answers.append([start_score,mean_score,answer, re.sub("Ġ"," ",''.join( par[par.index("</s>")+1:]))])
    return sorted(answers, key=itemgetter(0), reverse=True)



""" Versione Precedente per l'estrazione della risposta che prende in input la domanda in formato str e un file contenente una o più pagine di wikipedia  """
def get_answer_old(question,wikipage_path, batch_size = 24):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    long= long_answers.LongAnswer()
    
    model = long_answers.LSTM().to(device)

    
    bert_model = RobertaModel.from_pretrained("roberta-base")
    
    wikitext = " ".join(open(wikipage_path).readlines())
    paragraphs =long.html_tag(wikitext)
    questions = len(paragraphs)*[question]
    long.create_features(questions, paragraphs,forward=True)
    data = long.forward_data
    #directory = os.getcwd()
    
    model=model.to(device)
    predizioni =[]
    for i in range(0, len(data),batch_size):
        embs= long.sentence_bert(bert_model, data.input_ids.values.tolist()[i:i+batch_size], data.attention_mask.values.tolist()[i:i+batch_size]).to(device)
        prediction = model.forward(embs)
        pred = [round(x[0]) for x in prediction.tolist()]
        predizioni += pred
    bests_indices = get_best_results(predizioni)
    
    bests_indices = [i for i, x in enumerate(predizioni) if x == 1]
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    short_answer = short_answers.ShortAnswers()
    short_answer.create_model("roberta")
    short_answer.load_model("/mnt/Volume/HLT/short_answer_data/models/roberta-base-squad2/pytorch_model.bin")
    
    model_short = short_answer.model
    
    tokenizer = short_answer.tokenizer

    answers = []
    score = [] 
    for index in bests_indices:
        scores = model_short.forward( torch.tensor([data.input_ids.values.tolist()[index]]), torch.tensor([data.attention_mask.values.tolist()[index]]))
        score.append(scores)
        start_index = torch.argmax(scores[0])
        end_index = torch.argmax(scores[1])
        
        start_score = float(scores[0][0][start_index])
        answer = re.sub("Ġ"," ",''.join( tokenizer.convert_ids_to_tokens(data.input_ids.values.tolist()[index][int(start_index)+1:int(end_index)+2])))
        answers.append([start_score,answer])
    return sorted(answers, key=itemgetter(0), reverse=True)
