import pandas as pd 
from ElasticIndex import ElasticIndex
import os


index = ElasticIndex()
index_name = "gnq_clean"




############################## A) All_GNQ #####################################


qa_db = "/path/to/gnq_qa.db"
all_path = os.getcwd() + "/Risultati_Retriver/All_GNQ"

try:
    os.makedirs(all_path)
    n_results = [1,3,5,10]
    for n in n_results:
        results, metrics = index.evaluate(index_name= index_name, qa_db=qa_db, n_results = n)
        results.to_csv(all_path + '/results_' + str(n) +'.csv')
except OSError:
    print ("Creation of the directory %s failed" % all_path)
else:
    print ("Successfully created the directory %s " % all_path) 




############################## A) Partial_GNQ #####################################

qa_partial = "/path/to/gnq_partial_qa.db"
partial_path = os.getcwd() + "/Risultati_Retriver/Partial_GNQ"

try:
    os.mkdir(partial_path)
    os.mkdir(partial_path + "/ner")
    os.mkdir(partial_path + "/static")
    os.mkdir(partial_path + "/contextual")
    os.mkdir(partial_path + "/simple")

    n_results2 = [1,3,5,7,9,11,13]
    for n in n_results2:
        # ner
        results_ner, metrics_ner = index.evaluate(index_name=index_name, qa_db=qa_partial, n_results = n, entity= True)
        results_ner.to_csv(partial_path +'/ner/results_ner' + str(n) +'.csv')
    
        # static
        results_static, metrics_static = index.evaluate(index_name=index_name, qa_db=qa_partial, n_results = n, entity= True, synonym='static', n_syns=2)
        results_static.to_csv(partial_path + '/static/results_static' + str(n) +'.csv')
    
        # contextual
        results_contextual, metrics_contextual = index.evaluate(index_name=index_name, qa_db=qa_partial, n_results = n, entity= True, synonym='contextual', n_syns=2)
        results_contextual.to_csv(partial_path + '/contextual/results_contextual' + str(n) +'.csv')
        
        # simple
        results_simple, metrics_simple = index.evaluate(index_name= index_name, qa_db=qa_partial, n_results = n)
        results_simple.to_csv(partial_path + '/simple/results_simple' + str(n) +'.csv')    
   
except OSError:
    print ("Creation of the directory %s failed" % partial_path)
else:
    print ("Successfully created the directory %s " % partial_path) 
