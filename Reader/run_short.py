#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:44:22 2021

@author: guido & giuseppe
"""
import short_answers
import os


directory = os.getcwd()

### Creo una nuova istanza della classe Short Answer
short_answer = short_answers.ShortAnswers()

### Converte i dati dal formato NQ a quello SQuaD v2 e li salva su file
short_answer.nq_to_squad(file_path = directory + "/simplified_nq/simplified-nq-train.jsonl",output_path = directory + "/short_answer_data/nquad_data/")

### Carica i dati generati in precedenza
short_answer.get_train_examples(folder = directory +"/short_answer_data/nquad_data/", max_doc=100)

### Converte i dati dal formato SQuaD v2  in features da fornire in input al modello
short_answer.convert_to_features()

### Inizializzo il modello
short_answer.create_model("roberta_squad")

### Eseguo il fine-tuning del modello il modello 
short_answer.fine_tuning(batch=1, output_path = directory+"/short_answer_data/models/final/prove/", epochs = 1)


"""
### Carico un modello già addestrato o fine-tuned
short_answer.load_model(directory+"/short_answer_data/models/final/pytorch_model.bin")

"""