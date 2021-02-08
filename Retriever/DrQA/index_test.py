#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 09:21:46 2020

@author: giuseppe
"""
import regex
import time
import sqlite3
from drqa import retriever
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import pandas as pd
from drqa.retriever import utils
import os
import pandas as pd 


db = "/home/giuseppe/Scrivania/HLT_Project/Retriver/Process_gnq/gnq_articles.db"
connection = sqlite3.connect(db, check_same_thread=False)
tfidf = "/home/giuseppe/Scrivania/HLT_Project/Retriver/DrQA/gnq_articles-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz"
ranker = retriever.get_class('tfidf')(tfidf_path=tfidf)
qa_db = "/home/giuseppe/Scrivania/HLT_Project/Retriver/Process_gnq/gnq_qa.db"

def get_doc_text(doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]


def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > 0:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)
        

def search(query, n_results=5):
    doc_names, doc_scores= ranker.closest_docs(query, k = n_results)

    results = []
    for i in range(len(doc_names)):
        result = {}
        result["score"] = doc_scores[i]
        result["title"] = doc_names[i]
        result["text"] =  utils.normalize(get_doc_text(doc_names[i]))
        results.append(result)
    
    return results


def calculate(qa_pair, n_results):
    results = []
    question = qa_pair['question']
    answer = utils.normalize(qa_pair['answer'])
    
  
    # execute query
    res = search(query= question, n_results=n_results)
    
    # calculate performance metrics from query response info
    
    binary_results = [int(answer.lower() in doc["text"].lower()) for doc in res]
    ans_in_res = int(any(binary_results))
    
    #Calculate average precision
    m = 0
    precs = []

    for i, val in enumerate(binary_results):
        if val == 1:
            m += 1
            precs.append(sum(binary_results[:i+1])/(i+1))
            
    ap = (1/m)*np.sum(precs) if m else 0
    
    rec = (question, answer, ans_in_res, ap)
    results.append(rec)
    return results



def generate_QA(qa_db):
    connection = sqlite3.connect(qa_db, check_same_thread=False)
    cursor = connection.cursor()
    for row in cursor.execute('SELECT question, answer FROM QA'):
        qa_pair = {}
        qa_pair['question'] =row[0]
        qa_pair['answer']=row[1]
        yield qa_pair
    cursor.close()



def evaluate_retriver(qa_db, n_results = 3, workers = 8): 
        
    start_time = time.time()
        
    calculatex=partial(calculate, n_results=n_results)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        risultati = (executor.map(calculatex, generate_QA(qa_db)))
    results = list(chain(*risultati))
    
    # format results dataframe
    cols = ['question', 'answer', 'answer_present', 'average_precision']
    results_df = pd.DataFrame(results, columns=cols)
    
    # format results dict
    metrics = {'Recall'+ str(n_results): results_df.answer_present.value_counts(normalize=True)[1],
               'Mean Average Precision': results_df.average_precision.mean()}
    print("--- %s seconds ---" % (time.time() - start_time))

    return results_df, metrics

##################################################################################

output_path = os.getcwd() + "/Risultati_Retriver"

try:
    os.mkdir(output_path)
    n_results = [1,3,5,10]
    results_all =[]
    for n in n_results:
        results, metrics = evaluate_retriver(qa_db=qa_db, n_results=n)
        results_all.append(results)
        results.to_csv(output_path + '/results_' + str(n) +'.csv')
except OSError:
    print ("Creation of the directory %s failed" % output_path)
else:
    print ("Successfully created the directory %s " % output_path) 





#####################################################################################à

