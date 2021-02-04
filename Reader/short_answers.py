#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:52:42 2020

@author: guido & giuseppe

"""

import json
import logging
import os
import torch
import random
import pandas as pd

from tqdm import tqdm
from itertools import islice
from transformers.data.processors import squad
from transformers import LongformerTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW

"""  Creo la classe ShortAnswers """

class ShortAnswers():
    def __init__(self):
        self.train = None
        self.test = None
        self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.processor = squad.SquadV2Processor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    """ Metodo per leggere più righe del file alla volta, utile per il multiprocessing """    
        
    def read_lines(self,path,i=0, step=20):
        lines=[]
        
        with open(path) as f:
            lines_gen = islice(f,i,i+step)
            for line in lines_gen:
                l = json.loads(line)
                lines.append(l)
        f.close()
        return lines    
       
    """ Rimuove gli HTML tags """
    
    def clean_text(self,start_token, end_token, doc_tokens, doc_bytes,
                   ignore_final_whitespace=True):
      text = ""
      
      #last_index = end_token if ignore_final_whitespace else end_token + 1
      for index in range(start_token, end_token):
        token = doc_tokens[index]
        if token["html_token"]:
          continue
        text += token["token"]+" "
        
      return text
    
    
    """Converte i Natural Questions record nel formato SQuAD"""

    def nq_record_to_squad(self,record):
    
      doc_bytes = record["document_text"].encode("utf-8")
      tokens = record["document_text"].split(" ")
    
      doc_tokens= []
      start_byte=0
      end_byte=0
      for i,token in enumerate(tokens):
          #if "<" in token:
          if i>0:
              start_byte = end_byte+1
          else:start_byte = end_byte
          end_byte += len(token)+1
          if "<" in token: html_token=True
          else: html_token=False
          doc_tokens.append({ "token": token, "start_byte": start_byte, "end_byte": end_byte, "html_token": html_token })
      
    
    
      # NQ training data has one annotation per JSON record.
      annotation = record["annotations"][0]
    
      short_answers = annotation["short_answers"]
      # Skip examples that don't have exactly one short answer.
      # Note: Consider including multi-span short answers.
      if len(short_answers) != 1:
        return
      short_answer = short_answers[0]
    
      long_answer = annotation["long_answer"]
      # Skip examples where annotator found no long answer.
      if long_answer["start_token"] == -1:
        return
      # Skip examples corresponding to HTML blocks other than <P>.
      long_answer_html_tag = doc_tokens[long_answer["start_token"]]["token"]
      if long_answer_html_tag != "<P>":
        return
    
      paragraph = self.clean_text(
          long_answer["start_token"], long_answer["end_token"], doc_tokens,
          doc_bytes)
      answer = self.clean_text(
          short_answer["start_token"], short_answer["end_token"], doc_tokens,
          doc_bytes)
      before_answer = self.clean_text(
          long_answer["start_token"], short_answer["start_token"], doc_tokens,
          doc_bytes, ignore_final_whitespace=True)
    
      return {"title": record["document_text"].split("-")[0][:-1],
              "paragraphs":
                  [{"context": paragraph,
                    "qas": [{"answers": [{"answer_start": len(before_answer),
                                          "text": answer}],
                             "id": record["example_id"],
                             "question": record["question_text"]}]}]}
          
    """ Converte il dataset  Natural Questions nel formato SQuAD """      
          
    def nq_to_squad(self,file_path,output_path,rows_at_time=5000): 
        records = 0
        nq_as_squad = {"version": "simplified", "data": []}
        with open(file_path) as file:    
            for line in file:
              records += 1
              nq_record = json.loads(line)
              squad_record = self.nq_record_to_squad(nq_record)
              
              if squad_record:
                nq_as_squad["data"].append(squad_record)
              if records % rows_at_time == 0:
                with open(output_path+"_"+str(records-rows_at_time)+"_"+str(records)+".jsonl", "w") as f:
                  json.dump(nq_as_squad, f)
                  del nq_as_squad
                  nq_as_squad = {"version": "simplified", "data": []}
                logging.info("processed %s records", records)
              #if records>5000:
              #    break
        print("Converted %s NQ records into %s SQuAD records." %
              (records, len(nq_as_squad["data"])))
        
        
    """ Carice il dataset già convertito """    
        
    def load_train(self,folder="/short_answer_data/nquad_data/"):
        folder_path = os.getcwd()+folder
        dataset=[]
        data=[]
        for filename in tqdm(os.listdir(folder_path)):
             with open(folder_path+"/"+filename) as file:
                line = file.readlines()[0]
                line= json.loads(line)
                data.append(line)
        for el in data:
            dataset += el["data"]
        self.dataset=dataset
        return dataset
    
    """ genera gli esempi utili per il training """
    
    def get_train_examples(self,folder="/short_answer_data/nquad_data/", max_doc = 100):
        path =  folder
        self.train_examples = []
        i = 0
        print("Getting training examples from a squad format")
        for filename in tqdm(os.listdir(path)):
            i += 1
            self.train_examples+=(self.processor.get_train_examples(path,filename))
            if i>max_doc:
                print("Loaded examples from " + str(i) + " documents")
                break
            

    """ Converte i dati caricati in feature da fornire al modello """     
        
    def convert_to_features(self):
        self.features = squad.squad_convert_examples_to_features(self.train_examples,tokenizer=self.tokenizer, max_seq_length=256, doc_stride=256, max_query_length=128,padding_strategy = "max_length" , is_training="train")
        #self.features = squad.squad_convert_examples_to_features(self.dataset,self.tokenizer, max_seq_length=1024, doc_stride=1024, max_query_length=128, is_training=True)
        self.training_features = random.choices(self.features, k=int(len(self.features)*0.75))
        self.validation_features = [i for i in self.features if i not in self.training_features]
        return self.features        
    
    
    """ Inizializza il modello """
    
    def create_model(self,transformer = "longformer"):
        if transformer == "distilbert":
            from transformers import DistilBertForQuestionAnswering
            self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        elif transformer == "bert":
            from transformers import BertForQuestionAnswering
            self.model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")    
        elif transformer == "roberta":
            from transformers import RobertaForQuestionAnswering
            self.model = RobertaForQuestionAnswering.from_pretrained("roberta-base")   
        elif transformer == "roberta_squad":
            from transformers import RobertaForQuestionAnswering
            self.model = RobertaForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        elif transformer == "longformer":
            from transformers import LongformerForQuestionAnswering
            self.model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-base-4096")                    
        elif transformer == "bart":
            from transformers import BartForQuestionAnswering
            self.model = BartForQuestionAnswering.from_pretrained("facebook/bart-base")    
        elif transformer == "electra":
            from transformers import ElectraForQuestionAnswering
            self.model = ElectraForQuestionAnswering.from_pretrained("google/electra-small-discriminator")       
        else:
            print("The model you chose is not available in this version. You can try to manually change the code or manually overwrite the variable self.model")             
            print("The available choices are 'distilbert' , 'bert' , 'roberta' , 'longformer' , 'bart' , 'electra' ")
    
    """ Esegue il fine-tuning """        
    
    def fine_tuning(self, optim = "adam_w", epochs = 3, batch = 12, output_path = os.getcwd() + "/short_answer_data/models/"):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model.to(device)
        self.model.train()
        training_features = {"input_ids" :[] , "attention_mask" : [],"start_position" : [], "end_position" : []}
        for el in self.features:
            training_features["input_ids"].append(el.input_ids)
            training_features["attention_mask"].append(el.attention_mask)
            training_features["start_position"].append(el.start_position)
            training_features["end_position"].append(el.end_position)
        self.training_features = pd.DataFrame(training_features)
        train_loader = DataLoader(training_features, batch_size=16, shuffle=True)
        if optim == "adam_w":
            self.optim = AdamW(self.model.parameters(), lr=5e-5)
        step = batch
        for epoch in range(epochs):
           for i in tqdm(range(0,len(self.training_features),step)):
                input_ids = torch.tensor(self.training_features.iloc[i:i+step].input_ids.values.tolist()).to(self.device)
                attention_mask = torch.tensor(self.training_features.iloc[i:i+step].attention_mask.values.tolist()).to(self.device)
                start_positions = torch.tensor(self.training_features.iloc[i:i+step].start_position.values.tolist()).to(self.device)
                end_positions = torch.tensor(self.training_features.iloc[i:i+step].end_position.values.tolist()).to(self.device)
                self.optim.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                self.optim.step()
                if i% 100*step == 0:
                    self.model.save_pretrained(output_path+str(i)+"model.pt")
           self.model.save_pretrained(output_path+str(i)+"epoch_"+str(epoch)+"model.pt")
        self.model.eval()
        
        
    """ Carica un modello pre-fine-tuned """    
        
    def load_model(self, path):
      if self.model:
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print ("Model loaded from the following folder : {}".format(path))
      else:
        print("Before calling this method you should create a model with the method 'create_model()' ")
    
    
    """ esegue un forward """
    
    def forward(self,input_ids,attention_mask):
        input_ids = torch.tensor(input_ids).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)
        model = self.model.to(self.device)
        output = model.forward(input_ids)[0]
        _, predicted = torch.max(output.data, 1)
        return predicted
    
    """ Valuta il modello  """
    
    def evaluate_model(self):
        correct = 0
        total = 0
        i=1
        self.model= self.model.to(self.device)
        for row in self.validation.iterrows():
            
            sent = torch.tensor([row[1].input_ids]).to(self.device)
            label = torch.tensor([row[1].labels]).to(self.device)
            output = self.model.forward(sent)[0]
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted.cpu() == label.cpu()).sum()
            del sent, label, output, predicted, _
            #gc.collect()
            torch.cuda.empty_cache()
            if i%10 == 0:
              accuracy = 100.00 * correct.numpy() / total
              print('Iteration: {}. Accuracy: {}%'.format(i,  accuracy))#, end='\r')
            i+=1
            
        accuracy = 100.00 * correct.numpy() / total
        print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, accuracy))#, end='\r')
   
