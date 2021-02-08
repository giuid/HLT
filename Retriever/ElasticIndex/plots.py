#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:50:16 2020

@author: omen
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as pl
import os

############# A) ALL #######################################################

all_path = os.getcwd() + "/Risultati_Retriver/All_GNQ"
n_results = [1,3,5,10]
results = []

for el in n_results:
    results.append(pd.read_csv(all_path+ '/results_' + str(el) +'.csv').sample(frac = 1) )
    

recall1 = []
recall3 =[]
recall5 = []
recall10 = []
map1 = []
map3= []
map5= []
map10= []




for i in range(1, len(results[0]), int(len(results[0])/10)):
    r1 = (sum(results[0].iloc[:i].answer_present) / len(results[0].iloc[:i].answer_present))
    recall1.append(r1)
    r3 = (sum(results[1].iloc[:i].answer_present) / len(results[1].iloc[:i].answer_present))
    recall3.append(r3)
    r5 = (sum(results[2].iloc[:i].answer_present) / len(results[2].iloc[:i].answer_present))
    recall5.append(r5)
    r10 = (sum(results[3].iloc[:i].answer_present) / len(results[3].iloc[:i].answer_present))
    recall10.append(r10)
    m1 = results[0].iloc[:i].average_precision.mean()
    map1.append(m1)
    m3 = results[1].iloc[:i].average_precision.mean()
    map3.append(m3)
    m5 = results[2].iloc[:i].average_precision.mean()
    map5.append(m5)
    m10 = results[3].iloc[:i].average_precision.mean()
    map10.append(m10)
    print(i)


#Recall
question = np.arange(0, len(results[0]), int(len(results[0])/10))

pl.figure(figsize=(13, 10.5))

pl.axis([1, len(results[0]), 0.6, 1]) 
pl.plot(question,recall1, label="1",linestyle='dashed')
pl.plot(question,recall3, label="3",linestyle='dashed')
pl.plot(question,recall5, label="5",linestyle='dashed')
pl.plot(question,recall10, label="10",linestyle='dashed')
pl.xlabel("Questions")
pl.ylabel("Recall")
pl.title('Corpus Size vs Recall (by Number Search Results) ')
#inserisce la griglia nel grafico
pl.grid()
#consente di salvare il plot in una directory
#Il comando plt.legend consente di inserire una legenda nel grafico e determinarne la posizione
#per i posizionamenti alternativi della legend e per il controllo delle sue ulteriori proprietà consultare help(pyplot.legend)
pl.legend(title = "Top N Doc", loc="upper right") 
pl.savefig(all_path + "/Recall.png")

#MAP
question = np.arange(0, len(results[0]), int(len(results[0])/10))

pl.figure(figsize=(13, 10.5))

pl.axis([0, len(results[0]), 0.6, 0.80]) 

pl.plot(question,map1, label="1")
pl.plot(question,map3, label="3")
pl.plot(question,map5, label="5")
pl.plot(question,map10,label="10")
pl.xlabel("Questions")
pl.ylabel("Mean Average Precision")
pl.title('Corpus Size vs MAP (by Number Search Results) ')
#inserisce la griglia nel grafico
pl.grid()
#consente di salvare il plot in una directory
#Il comando plt.legend consente di inserire una legenda nel grafico e determinarne la posizione
#per i posizionamenti alternativi della legend e per il controllo delle sue ulteriori proprietà consultare help(pyplot.legend)
pl.legend(title = "Top N Doc", loc="upper right")
pl.savefig(all_path + "/Mean Average Precision.png")


############# B) PARTIAL #######################################################

partial_path = os.getcwd() + "/Risultati_Retriver/Partial_GNQ"
n_results = [1,3,5,7,9,11,13]
path = ["simple", "static", "ner", "contextual"]
results_partial = []

for st in path:
    for el in n_results:
        results_partial.append([pd.read_csv(partial_path +"/"+ st +'/results_' +st + str(el) +'.csv'), st, el ])
    
ner_recall = []
static_recall = []
contextual_recall = []
simple_recall = []

ner_map = []
static_map = []
contextual_map = []
simple_map = []



for el in results_partial:
    r = el[0].answer_present.value_counts(normalize=True)[1]
    if (el[1]=="simple"):
        simple_recall.append(r)
    elif (el[1]=="static"):
        static_recall.append(r)
    elif (el[1]=="ner"):
        ner_recall.append(r)
    else:
        contextual_recall.append(r)
    
        
for el in results_partial:
    a = el[0].average_precision.mean()
    if (el[1]=="simple"):
        simple_map.append(a)
    elif (el[1]=="static"):
        static_map.append(a)
    elif (el[1]=="ner"):
        ner_map.append(a)
    else:
        contextual_map.append(a)        


#Recall
#pl.figure(figsize=(13, 10.5))

pl.axis([0, 13, 0.63, 0.95])
pl.plot(n_results,simple_recall, label="No Expansion")
pl.plot(n_results,ner_recall, label="Entity Enrichment")
pl.plot(n_results,static_recall, label="Static Expansion")
pl.plot(n_results,contextual_recall, label="Contextual Expansion")
pl.xlabel("Top N Docs")
pl.ylabel("Recall")
pl.title('Number of Documents Retrived vs Recall by Query Expansion Method ')
#inserisce la griglia nel grafico
pl.grid()
#consente di salvare il plot in una directory
#Il comando plt.legend consente di inserire una legenda nel grafico e determinarne la posizione
#per i posizionamenti alternativi della legend e per il controllo delle sue ulteriori proprietà consultare help(pyplot.legend)
pl.legend(title = "Expansion Type", loc="lower right")
pl.savefig(partial_path + '/Recall.png')


#MAP
#pl.figure(figsize=(13, 10.5))

pl.axis([0, 13, 0.63, 0.78])
pl.plot(n_results,simple_map, label="No Expansion")
pl.plot(n_results,ner_map, label="Entity Enrichment")
pl.plot(n_results,static_map, label="Static Expansion")
pl.plot(n_results,contextual_map, label="Contextual Expansion")
pl.xlabel("Top N Docs")
pl.ylabel("Mean Average Precision")
pl.title('Number of Documents Retrived vs MAP by Query Expansion Method ')
#inserisce la griglia nel grafico
pl.grid()
#consente di salvare il plot in una directory
#Il comando plt.legend consente di inserire una legenda nel grafico e determinarne la posizione
#per i posizionamenti alternativi della legend e per il controllo delle sue ulteriori proprietà consultare help(pyplot.legend)
pl.legend(title = "Expansion Type", loc="upper right")
pl.savefig(partial_path +'/MAP.png')



