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
output_path = os.getcwd() + "/Risultati_Retriver"




############# A) ALL #######################################################


n_results = [1,3,5,10]
results = []

for el in n_results:
    results.append(pd.read_csv(output_path+ '/results_' + str(el) +'.csv').sample(frac = 1) )
    

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

#pl.figure(figsize=(13, 10.5))

pl.axis([1, len(results[0]), 0.4, 1]) 
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
pl.savefig(output_path + "/Recall.png")


#MAP
question = np.arange(0, len(results[0]), int(len(results[0])/10))

#pl.figure(figsize=(13, 10.5))

pl.axis([0, len(results[0]), 0.4, 1]) 

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
pl.savefig(output_path + "/Mean Average Precision.png")


