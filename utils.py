#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:01:05 2020

@author: guido
"""

import pandas as pd
import json
import os
from textblob import TextBlob

PATH = os.getcwd()


def load_json(path):
    with open(path) as jsonFile:
        jsonel = json.load(jsonFile)
        jsonFile.close()
    return jsonel    

def convert_column(path,df_col):
    try:
        df = pd.read_csv(path)
        print("There was already a file in the directory")
    except: 
       # df_col = pd.DataFrame(df_col)
        df = pd.DataFrame()
        i=0
        for el in df_col:
            if type(el)== str:
                df = df_col
                break
            elif type(el) == dict:
                indexes = pd.DataFrame([df_col.index.values[i]]*len(el), columns=["id"])
                args = pd.DataFrame.from_dict(el, orient='index')
                args= args.reset_index()
                args = pd.concat([indexes,args],axis=1)
            elif type(el) == list:
                indexes = pd.DataFrame([df_col.index.values[i]]*len(el), columns=["id"])
                args = pd.DataFrame(el)
                args = pd.concat([indexes,args],axis=1)
            df =pd.concat([df,args])
            i+=1
        df.to_csv(path)
    return df

def convert_df(path,df):
    lista =[]
    for col in list(df.columns):
        df_out = convert_column(path+col,df[col])
        lista.append(df_out)
    return lista                

def load_squad(path):
    try:
        squad = pd.read_csv(path+ "/train-v2.0.csv")
        print("There was already a file in the directory")
    except:
        train = load_json(path + "/train-v2.0.json")
        data = train["data"]
        squad = pd.DataFrame()
        for el in data:
            paragraphs = el["paragraphs"]
            paragrapphi = pd.DataFrame()
            title = [el["title"]]
            for il in paragraphs:
                context = il["context"]
                context = pd.DataFrame([context],columns=["context"])
                qas = il["qas"]
                qassi = pd.DataFrame()
                for ul in qas:
                    if len(ul) != 5:
                        cal = pd.DataFrame.from_dict(ul)
                        cal = pd.concat([cal,context],axis=1)
                        text = cal.answers[0]["text"]
                        answer_start = cal.answers[0]["answer_start"]
                        answer = pd.DataFrame({'answer_text': [text], 'answer_start':  [answer_start], 
                                               "plausible_answer_text": [None], "plausible_answer_start": [None]})
                        cal = cal.drop(["answers"],axis=1)
                        cal= pd.concat([cal,answer], axis=1)
                        qassi=pd.concat([qassi,cal])
                         #print (cal)
                    if len(ul) == 5:
                       # print (ul)
                        al = pd.DataFrame({ key:pd.Series(value) for key, value in ul.items() })
                        al = pd.concat([al,context],axis=1)
                        #text = al.answers[0]["text"]
                        #answer_start = al.answers[0]["answer_start"]
                        answer = pd.DataFrame({'answer_text': [None], 'answer_start':  [None], 
                                               'plausible_answer_text': [al.plausible_answers[0]["text"]],'plausible_answer_start': [al.plausible_answers[0]["answer_start"]]})
                        al = al.drop(["answers","plausible_answers"],axis=1)
                        al = pd.concat([al,answer], axis=1)
                        qassi=pd.concat([qassi,al])
                paragrapphi = pd.concat([paragrapphi,qassi])
            title = pd.DataFrame([el["title"]]*len(paragrapphi), columns=["title"])
            title.reset_index(drop=True, inplace=True)
            paragrapphi.reset_index(drop=True, inplace=True)
            paragrapphi = pd.concat([title,paragrapphi], axis =1)
            squad = pd.concat([squad,paragrapphi])
        squad.to_csv(path+ "/train-v2.0.csv")
    return squad.fillna("")


def squad_passage_retrieval(context, answer_start,answer_length):
        sent = TextBlob(context)
        sentences= sent.sentences
        length = 0 
        ans_index = 0
        for el in sentences:
            length += len(el)+1
            if length >= answer_start:
                passage = str(el)
                ans_start = int(answer_start - (length-len(el)))
                answer_end = int(answer_start+answer_length)
                ans_end = int(ans_start+answer_length)
                break
            ans_index+=1
    
        sent_num = len(sentences)
        df = pd.DataFrame([[answer_end,passage,ans_start,ans_end,sent_num,ans_index]],columns=["answer_end","passage","ans_start","ans_end","sent_num","ans_index"])
        #df.to_csv(PATH+"/passages.csv")
        return df
    
    
def squad_passage(squad):
    try:
        df = pd.read_csv(PATH+"/passages.csv")
        print("There was already a file 'passages.csv' in the directory "+ PATH )
    except:
        df = pd.DataFrame(columns = ["passage","ans_start","sent_num","ans_index"])
        for context in squad.context.unique():
            for index, row in squad[squad.context == context].iterrows():
                if type(row[7]) == str:
                     info = squad_passage_retrieval(context,row[9],len(row[8]))
                else:
                    info = squad_passage_retrieval(context,row[7],len(row[6]))
                df = df.append(info)
        df.to_csv(PATH+"/passages.csv")
    return df

def wrong_passages(context):
    wrong_passages = []
    for cont in context:
       passages = TextBlob(cont)
       wrong_passages.append(passages.sentences)
    return wrong_passages        

def squad_preprocessing(path):
    squad = load_squad(PATH)
    passages = squad_passage(squad)
    squad = pd.concat([squad, passages],axis=1)
    return squad