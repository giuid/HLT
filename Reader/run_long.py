#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:57:48 2020

@author: guido
"""

import long_answers
import os


directory = os.getcwd()

""" Nuova istanza della classe LongAnswer """
long = long_answers.LongAnswer()

""" Preprocess the Simplified Nq to create a dataset useful for the training"""
long.build_dataset(file_path = f"{directory}/simplified_nq/simplified-nq-train.jsonl", output_path = f"{directory}/long_answer_data/long_answer_train.csv")

""" Se già prodotto carica il dataset utile per l'addestramento """
#long.load_train(f"{directory}/long_answer_data/long_answer_train_no_paragraph.csv")

""" Create Input ids, attention mask and labels in order to train the model"""
long.create_features()

""" inizializza un nuovo modello """
long.create_model()

""" Addestra il modello """
long_answers.train(long.model, long.training_data, long.validation, epochs=2)


        
