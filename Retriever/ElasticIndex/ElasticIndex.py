#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:15:27 2020

@author: omen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:28:02 2020

@author: omen
"""
import json
from elasticsearch import Elasticsearch
import tqdm
from tqdm.notebook import tqdm as tq
from elasticsearch.helpers import parallel_bulk
import time
import numpy as np
import pandas as pd
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from elasticsearch_dsl import Q, Search
from functools import partial
from QueryExpander import QueryExpander
import sqlite3

class ElasticIndex:
    
    def __init__(self):
        config = {'host':'localhost', 'port':9200}
        client = Elasticsearch([config])
        print(client.ping())
        self.client = client
        self.query = None
        self.query_type = None
        
    def create(self, index_name):
        #Creates an index in Elasticsearch if one isn't already there.
        
        if(self.client.indices.exists(index_name)):
            print("This index already exists")
        else:
            return self.client.indices.create(
                index=index_name,
                body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "stop_stem_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "char_filter": [
                                        "html_strip"
                                    ],
                                    "filter":[
                                        "lowercase",
                                        "stop",
                                        "porter_stem"
                                    ]
                                    
                                }
                            }
                        }
                    },
                    "mappings": {
                        "dynamic": "strict", 
                        "properties": {
                            "document_title": {"type": "text", "analyzer": "stop_stem_analyzer"},
                            "document_text": {"type": "text", "analyzer": "stop_stem_analyzer"}
                            }
                        }
                    },
                ignore=400,
            )
 
    def delete(self, index_name):
        if not(self.client.indices.exists(index_name)):
            print("This index does not exist")
        else:
            return self.client.indices.delete(index_name)

  

    def generate_docs(self, corpus):
        connection = sqlite3.connect(corpus, check_same_thread=False)
        cursor = connection.cursor()
        for row in cursor.execute('SELECT * FROM documents'):
            line = {}
            line['document_title'] =row[0]
            line['document_text']=row[1]
            yield line
        cursor.close()
        
    
    def populate(self, index_name, corpus, thread =4, chunk =500):
        start_time = time.time()
        connection = sqlite3.connect(corpus, check_same_thread=False)
        cursor = connection.cursor()
        number_of_docs = 0
        for row in cursor.execute('SELECT * FROM documents'):
            number_of_docs+=1
        
        cursor.close()
        
        
        if not(self.client.indices.exists(index_name)):
            print("This index does not exist")
        else:
            
            print("Creating an index...")
            #create_index(self.client, index_name)
        
            print("Indexing documents...")
            progress = tqdm.tqdm(unit="docs", total=number_of_docs)
            successes = 0
            for ok, action in parallel_bulk(
                client=self.client, index=index_name, actions=self.generate_docs(corpus), thread_count=thread, chunk_size=chunk
            ):
                progress.update(1)
                successes += ok
            print("Indexed %d/%d documents" % (successes, number_of_docs))
            print("--- %s seconds ---" % (time.time() - start_time))



    def search(self, index_name, question_text, n_results = 3, entity=False, synonym=None, n_syns = 2):
            
            if not entity and not synonym:
                self.query_type ='Normal Query'
                self.query = QueryExpander(question_text).get_query()
            elif entity and not synonym:
                self.query_type ='NER Query'
                self.query = QueryExpander(question_text, entity = True).get_query()
            elif (entity and synonym =='static'):
                self.query_type ='Static Query'
                self.query = QueryExpander(question_text, entity = True, synonym = 'static', n_syns = n_syns).get_query()
            elif (entity and synonym =='contextual'):
                self.query_type ='Contextual Query'
                self.query = QueryExpander(question_text, entity = True, synonym = 'contextual', n_syns = n_syns).get_query()

            s = Search(using=self.client, index=index_name)
            s = s[0:n_results]
            s = s.query(self.query)
            response = s.execute()
            return response.to_dict()

    def generate_QA(self, qa_db):
        connection = sqlite3.connect(qa_db, check_same_thread=False)
        cursor = connection.cursor()
        for row in cursor.execute('SELECT question, answer FROM QA'):
            qa_pair = {}
            qa_pair['question'] =row[0]
            qa_pair['answer']=row[1]
            yield qa_pair
        cursor.close()


    def calculate(self, qa_pair, index_name, n_results, entity, synonym, n_syns ):
        results = []
        question = qa_pair['question']
        answer = qa_pair['answer']
        
      
        # execute query
        res = self.search(index_name = index_name, question_text= question, n_results=n_results, entity=entity, synonym=synonym, n_syns = n_syns)
        
        # calculate performance metrics from query response info
        duration = res['took']
        binary_results = [int(answer.lower() in doc['_source']['document_text'].lower()) for doc in res['hits']['hits']]
        ans_in_res = int(any(binary_results))
        
        #Calculate average precision
        m = 0
        precs = []
    
        for i, val in enumerate(binary_results):
            if val == 1:
                m += 1
                precs.append(sum(binary_results[:i+1])/(i+1))
                
        ap = (1/m)*np.sum(precs) if m else 0
            
        rec = (question, answer, duration, ans_in_res, ap)
        results.append(rec)
        return results
    
    
    
    def evaluate(self, index_name, qa_db, n_results = 3, entity=False, synonym=None, n_syns = 2, workers = 8): 
        
        start_time = time.time()
        
        calculate_partial=partial(self.calculate, index_name=index_name, n_results=n_results, entity=entity, synonym=synonym, n_syns=n_syns)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            risultati = executor.map(calculate_partial, self.generate_QA(qa_db))        
        results = list(chain(*risultati))
        
        # format results dataframe
        cols = ['question', 'answer', 'query_duration', 'answer_present', 'average_precision']
        results_df = pd.DataFrame(results, columns=cols)
        
        # format results dict
        metrics = {'Recall'+ str(n_results): results_df.answer_present.value_counts(normalize=True)[1],
                   'Mean Average Precision': results_df.average_precision.mean(),
                   'Average Query Duration':results_df.query_duration.mean()}
        print("--- %s seconds ---" % (time.time() - start_time))
    
        return results_df, metrics

    def get_query(self):
        return self.query
    
    def get_query_type(self):
        return self.query_type

    def get_info(self, index_name):
        if not(self.client.indices.exists(index_name)):
            print("This index does not exist")
        else:
            return self.client.indices.get(index_name)
        
    def docs_count(self, index_name):
       if not(self.client.indices.exists(index_name)):
           print("This index does not exist")
       else:
           return self.client.cat.count(index_name)
       
    def show_all(self):
        for index in self.client.indices.get('*'):
            print (index)


###########################################################################################################################



# POPULATE INDEX
'''
db = "/path/to/gnq_articles.db"
index = ElasticIndex()
index_name = "gnq_clean"
index.create(index_name)
index.populate(index_name= index_name, corpus = db, thread =4, chunk =100)

'''



