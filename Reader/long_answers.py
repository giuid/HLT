#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:19:55 2020

@author: guido
"""
import torch

import json
import multiprocessing
import psutil
import gc
import ast
import csv
import re
import os
import pandas as pd
import numpy as np
from itertools import islice
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch.nn as nn

""" Definisco il device sul quale eseguire le operazioni """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" Salva un chekpoint del modello in fase di training """

def save_checkpoint(save_path, model, optimizer, valid_loss):

        if save_path == None:
            return
        
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'valid_loss': valid_loss}
        
        torch.save(state_dict, save_path)
        print(f'Model saved to ==> {save_path}')

""" Carica un chekpoint del modello """

def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']

""" Salva le metriche del modello """

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

""" Carica le metriche del modello """

def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

""" Ottiene l'accuracy """

def get_accuracy( validation_ids,attention_mask,labels, model,bert_model):
        falsi_positivi=0
        falsi_negativi=0
        true_positive=0
        true_negative=0
        predizioni = []
        model = model.to(device)
        for i in tqdm(range(0,len(validation_ids),24)):
            embs= sentence_bert(bert_model, validation_ids[i:i+24], attention_mask[i:i+24]).to(device)
            prediction = model.forward(embs)
            predic = [round(x[0]) for x in prediction.tolist()]
            predizioni += predic
        
        for i in range(len(predizioni)):
            if predizioni[i] == labels[i] == 1:
                true_positive+=1
            elif predizioni[i] == labels[i] == 0:
                true_negative+=1
            elif predizioni[i]==0:
                falsi_negativi+=1
            else:
                falsi_positivi+=1
                
        return (true_negative+true_positive)/(true_negative+true_positive+falsi_negativi+falsi_positivi)

""" Genera gli embedding con un qualsiasi modello Transformer """

def sentence_bert( model,input_idss, attention_masks):
    list_of_emb=[]
    cuda = torch.device('cuda')
    model = model.to(cuda)
    for ids, a_mask in (zip(input_idss, attention_masks)):
         # long input document
        input_ids = torch.tensor([ids], device=cuda)  # batch of size 1

        attention_mask = torch.tensor([a_mask], device=cuda) # initialize to local attention
 
        with torch.no_grad():

            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)
        token_vecs_cat = []

        for token in token_embeddings:
            cat_vec = torch.stack((token[-1], token[-2], token[-3], token[-4]), dim=0).sum(dim=0)
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec.to("cpu"))
          
                
                
        list_of_emb.append(torch.stack(token_vecs_cat))
    list_of_emb= torch.stack(list_of_emb)
    return list_of_emb    

""" Addestra il modello """

def train(model,training_data, validation_data, batch_size = 24, epochs=3, file_path = os.getcwd()+"/long_answer_data/models"):
    from transformers import RobertaModel
    bert_model = RobertaModel.from_pretrained("roberta-base")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_valid_loss = float("Inf")
    criterion = nn.BCELoss() 
    
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)

    eval_every =24000
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    model = model.to(device)
    model.train(),

    
    training_ids = training_data.input_ids.values.tolist()
    training_attention_mask = training_data.attention_mask.values.tolist()
    training_labels = training_data.labels.values.tolist()
    validation_ids = validation_data.input_ids.values.tolist()
    validation_attention_mask = validation_data.attention_mask.values.tolist()
    validation_labels = validation_data.labels.values.tolist()
    print("\nStarting Training: \n")
    for epoch in range(epochs):
        for i in tqdm(range(0,len(training_data),batch_size)):
            emb = sentence_bert(bert_model, training_ids[i:i+batch_size], training_attention_mask[i:i+batch_size])
            emb = emb.to(device)
            lab = torch.tensor(training_labels[i:i+batch_size]).to(device).unsqueeze(1)
            output = model(emb)
            #output, indexes = torch.max(outputs, dim=1)
            loss = criterion(output, lab.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            global_step += 1
            
            if global_step % int(eval_every/batch_size) == 0:
                print ("\nEvaluation : \n")
                model.eval()
                with torch.no_grad():  
                    
                    for i in tqdm(range(0,len(validation_data),batch_size)):
                        emb = sentence_bert(bert_model, validation_ids[i:i+batch_size], validation_attention_mask[i:i+batch_size])
                        emb = emb.to(device)
                        lab = torch.tensor(validation_labels[i:i+batch_size]).to(device).unsqueeze(1)
                        output = model(emb)
                        #output, indexes = torch.max(outputs, dim=1)

                        loss = criterion(output, lab.float())
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / (eval_every/batch_size)
                average_valid_loss = valid_running_loss / len(validation_data)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()
                
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch+1, epochs, global_step, epochs*len(training_data),average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
        

""" La classe LSTM per la creazione del modello """

class LSTM(nn.Module):
     def __init__(self, dimension=128):
        super(LSTM, self).__init__()

        self.dimension = dimension
        self.lstm = nn.LSTM(768, dimension, 1, bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(2*dimension, 1)
        
     def forward(self,bert_embeddings):
        output,_ = self.lstm(bert_embeddings)
        dropped = self.drop(output)[:, -1, :]
        prediction = self.fc(dropped)
        #prediction = torch.cat(prediction,dim=1)
        sigmoid = nn.Sigmoid()
        pred = sigmoid(prediction)
        return pred

""" La classe LongAnswer """

class LongAnswer():

    ## Initialize all the object variables ##
    def __init__(self,):
        self.train = None
        self.text = None
        self.question = None
        self.labels = None
        self.test = None
        self.input_ids =None
        self.attention_mask = None
        self.validation = None
        self.training_data = None
        self.tokenizer = None#LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = torch.nn.MSELoss()
        self.model = None


        self.optimizer = None
        self.learning_rate = 1e-05


    """ Permette di scegliere il tipo di tokenizer """
    
    def set_tokenizer(self, tokenizer = "roberta"):
        if tokenizer == "longformer":
            from transformers import LongformerTokenizer
            self.tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
            self.tokenizer_type = tokenizer
        elif tokenizer == "roberta":
            from transformers import RobertaTokenizer
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.tokenizer_type = tokenizer
        elif tokenizer == "bert":
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.tokenizer_type = tokenizer
        else:
            print("Error, the tokenizers allowed are 'longformer' , 'roberta' , 'bert' ")

    
    """ Define a function to read multiple lines at a time from a file """
    def read_lines(self,path,i=0, step=20):
        lines=[]

        with open(path) as f:
            lines_gen = islice(f,i,i+step)
            for line in lines_gen:
                l = json.loads(line)
                lines.append(l)
        f.close()
        return lines
    

    """## Define a function to recognize and ectract paragraphs (sentences in <P></P>) ##"""
    def html_tag(self,text):
        soup = BeautifulSoup(text,'lxml')
        paragraphs=[]
        par = soup.find_all('p')
        if par :
            for el in par:
                if el != " ":
                 paragraphs.append(el.text)
            paragraphs = [re.sub(r'\[.*\]', '', i) for i in paragraphs]     
            return paragraphs
        else: return []


    """ ## Define a function to preprocess the lines of the nq_simplified ##"""
    def preprocess_line(self,line,sampling_rate=15):
        #context = line['document_text']
        #paragraphs = self.html_tag(line['document_text'])
        text = line['document_text'].split(' ')
        question = line['question_text']
        annotations = line['annotations'][0]
        processed_rows=[]

        for i, candidate in enumerate(line['long_answer_candidates']):
            label = i == annotations['long_answer']['candidate_index']
            if (label or (i % sampling_rate == 0)):
                start = candidate['start_token']
                end = candidate['end_token']
                txt = self.html_tag(" ".join(text[start:end]))


                if txt:
                    processed_rows.append({
                        #'context': context,
                        'text': txt[0],
                        'is_long_answer': label,
                        'question': question,
                        'annotation_id': annotations['annotation_id'],
                        #'paragraphs' : paragraphs
                    })
            else: continue
        return processed_rows


    """effettua il preprocessing del test set di NQ"""
    def preprocess_test_line(self,line,sampling_rate=15):
        context = line['document_text']
        #paragraphs = self.html_tag(line['document_text'])
        text = line['document_text'].split(' ')
        question = line['question_text']
        #annotations = line['annotations'][0]
        processed_rows=[]

        for i, candidate in enumerate(line['long_answer_candidates']):
            #label = i == annotations['long_answer']['candidate_index']
            #if (label or (i % sampling_rate == 0)):
            start = candidate['start_token']
            end = candidate['end_token']
            txt = self.html_tag(" ".join(text[start:end]))


            if txt:
                processed_rows.append({
                    'context' : context,
                    'text': txt[0],
                    'question': question,
                    'example_id': line["example_id"],
                })
            #else: continue
        return processed_rows
    
    
    """## define a function to extract data from the multiprocessing output in build_dataset() ##"""
    
    def extract_data(self,data):
        dataset =[]
        for el in data:
                if el:
                   dataset += (el)
        return dataset

    def _compute_labels(self,is_right_answer):
        labels = []
        for el in is_right_answer:
                if el:
                    labels.append(1)
                else:
                    labels.append(0)
        return labels
        
    """## Define a function that convert the nq_train to a format useful for the sentence classification ##"""
    
    def build_dataset(self,file_path="/home/guido/HLT/simplified_nq/simplified-nq-train.jsonl",output_path="/home/guido/HLT/long_answer_data/long_answer_train.csv", n_rows=2000000,step=200, sampling_rate=15, test = False):
        open(output_path ,'w').close()
       # processed_rows = []
        dataset=[]
        for i in tqdm(range(0,n_rows,step)):
            lines = self.read_lines(file_path,i,step)
            if not(lines):
                break
            p = multiprocessing.Pool(processes=8)
            if test:
                data = list(p.map(self.preprocess_test_line, lines))
            else:
                data = list(p.map(self.preprocess_line, lines))
            p.close()

            dataset+=self.extract_data(data)
            if psutil.virtual_memory().percent >90:

                dataset = pd.DataFrame(dataset)
                dataset.to_csv(output_path ,mode='a', sep=";", header=False)
                del dataset
                dataset=[]
                gc.collect()
            #if i>0 and 5000 % i == 0: print ("elaborated %s rows" % i )


        if dataset:
            dataset = pd.DataFrame(dataset)
            dataset = dataset[dataset['text'] != " "]
            dataset.to_csv(output_path ,mode='a', sep=";", header=False)
        if test:
            dataset = pd.read_csv(output_path, index_col=0,  names=["index","context","text","question","id"], delimiter=";",)#,"paragraphs"])
            self.test_text = list(dataset.text)
            self.test_question = list(dataset.question)
            self.test_dataset = dataset
        else:
            dataset = pd.read_csv(output_path, index_col=0,  names=["index","text","is_right_answer","question","id"], delimiter=";",)#,"paragraphs"])
            self.text = list(dataset.text)
            #self.labels = list(dataset.is_right_answer)
            self.question = list(dataset.question)
            self.labels = self.compute_labels(self.dataset.is_right_answer)
            self.dataset = dataset
        print("Converted {} and saved to the following path {}".format(file_path,output_path))
        
        
    """## Define a function that loads the already converted dataset (see the function build_dataset) ##"""
    
    def load_train(self,path):
        try:
            self.dataset = pd.read_csv(path, index_col=0,  names=["index","text","is_right_answer","question","id"])#,"paragraphs"])
            
            false = self.dataset[self.dataset["is_right_answer"]==False]
            true = self.dataset[self.dataset["is_right_answer"]==True]
            
            false_bal = false.sample(frac=(len(true)/len(false)))
            self.dataset = pd.concat([false_bal,true]).sample(frac=1)
            del true, false, false_bal
            self.text = list(self.dataset.text)
            self.question = list(self.dataset.question)
            self.labels = self._compute_labels(self.dataset.is_right_answer)
            print("Train loaded")
            print ( " The dataset was autamatically balanced with respect to the labels")

            #return self.train
        except:
            print("Error loading training data")
        

    """##Define a function that create the features to be fed to the model, from the converted dataset ##"""
    
    def create_features(self, question = None, text = None, tokenizer = None, max_length = 1024 ,padding= "max_length", test=False, forward=False, create_test = False):
        if test:
            if question is None:
                question = self.test_question
            if text is None:
                text = self.test_text
        else:
            if question is None:
                question = self.question
            if text is None:
                text = self.text 
        if tokenizer is None:
            self.set_tokenizer()
        else:
            self.set_tokenizer(tokenizer)
        if self.tokenizer_type in ["roberta","bert"] and max_length> 512:
            max_length = 512
            print("The maximum length of the embedding was set to {} because the selected tokenizer is {}".format(max_length,self.tokenizer_type))
        features = self.tokenizer(question,text,padding="max_length", max_length = max_length,  truncation=True)
        self.input_ids = features["input_ids"]
        self.attention_mask = features["attention_mask"]
        if forward:
            self.forward_data = pd.DataFrame({"input_ids" : self.input_ids,"attention_mask": self.attention_mask})

        else:    
            training_data = pd.DataFrame({"input_ids" : self.input_ids,"attention_mask": self.attention_mask, "labels" : self.labels})
            if create_test:
                self.training_data = training_data.sample(frac=0.65,random_state=200).reset_index(drop=True)
                temp = training_data.drop(self.training_data.index).reset_index(drop=True)
                self.validation  = temp.sample(frac=0.60,random_state=200).reset_index(drop=True)
                self.test = temp.drop(self.validation.index).reset_index(drop= True)
                print("Features Created, 65% training set, 20% validation set, 15% test set")
            else:
                self.training_data = training_data.sample(frac=0.75,random_state=200).reset_index(drop=True)
                self.validation = training_data.drop(self.training_data.index).reset_index(drop=True)
                print("Features Created, 75% training set, 25% validation set,")


    """## Define a function that loads the features already computed ##"""
    def load_features(self,path):
        training_data = pd.read_csv(path, index_col=0, delimiter = ";")
        self.training_data = training_data.sample(frac=0.75,random_state=200).reset_index(drop=True)
        self.validation = training_data.drop(self.training_data.index).reset_index(drop=True)
        print("Features loaded")


    """## If the features loaded are imported as strings you can convert them with the following function ##"""
    def features_to_csv(self,output_folder):
        try:
            training = pd.concat([self.training_data,self.validation])
            training.to_csv(output_folder+ '/training_data.csv', sep=";")
            #self.validation.to_csv(output_folder+ '/validation_data.csv', sep=";")
        except:
            print("Error while writing features to " + output_folder+ '/training_data.csv')

    """ Carica un modello salvato in precedenza """
    def load_model(self, path):
      if self.model:
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
      else:
        print("Before calling this method you should create a model with the method 'create_model()' ")
    
    """ Valuta il modello """
    def evaluate_model(self):

        losses = []

        correct_predictions = 0
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        with torch.no_grad(): 
            for row in self.validation.iterrows():
               
                sent = torch.tensor([row[1].input_ids]).to(self.device)
                attention_mask = torch.tensor([row[1].attention_mask]).to(self.device)
                label = torch.tensor([row[1].labels]).to(self.device)
                label = label.unsqueeze(1)#.to(torch.float32)
                output = self.model(sent,attention_mask)

                _, preds = torch.max(output, dim=1)

                loss = self.loss_fn(output, label[0])
          
                correct_predictions += torch.sum(preds == label)
          
                losses.append(loss.item())
                del sent, label, output#, _
                gc.collect()
                torch.cuda.empty_cache()
       
        return correct_predictions.double() / len(self.validation), np.mean(losses)  ,losses      


    """ Esegue un forward """

    def forward(self,question,document):
            outputs=[]
            predictions = []
            #losses = []
            if self.model:
                paragraphs = self.html_tag(document)
                self.model.to(self.device)
                for el in paragraphs:
                  with torch.no_grad():  
                    self.create_features(question,el,forward=True)
                    input_ids = torch.tensor([self.forward_data.input_ids.values.tolist()]).to(self.device)
                    attention_mask  = torch.tensor([self.forward_data.attention_mask.values.tolist()]).to(self.device)
    
                    output = self.model(input_ids, attention_mask)
                    outputs.append(output.to("cpu"))
    
                    _, prediction = torch.max(output, dim=1)
                    predictions.append(prediction)
                    del output, prediction
                    torch.cuda.empty_cache()
            return outputs, predictions


    """ Inizializza un modello """
    
    def create_model(self,dimension = 128):
        self.model = LSTM(dimension)
    
    """ permette di scrivere liste su file, anche liste annidate  """
    
    def write_output(self,output_file,lista,annidated=False):
     with open(output_file, "a",newline="") as file:
             wr = csv.writer(file)
             if annidated:
                 for el in lista:
                     wr.writerows(el)
             else: wr.writerows(lista)
             file.close()

    """ permette di leggere liste da file, anche liste annidate  """

    def read_output(self,output_file, numerical= True):
        with open(output_file, newline='') as f:
           # reader = csv.reader(f, delimiter= ",")
            if numerical:
                data = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            else:
                data = [[x for x in rec] for rec in csv.reader(f, delimiter=',')]
            f.close()
            return data

    """ Converte liste salvate in formato stringa in list """

    def convert_line_to_list(self, line):
        lista = line.apply(ast.literal_eval)
        return list(lista)
    

    """ Converte un dataset salvato in formato stringa in dataframe """

    def convert_str_to_list(self, output_path , step =800):
        dataframe = self.training_data
        training_data_dict = {}
        written = False
        for col in dataframe.columns:
            datas = []
            open(output_path + str(col)+ ".csv", "w" ).close()
            if type(dataframe[col][0]) is str:
                for i in tqdm(range(0,len(dataframe),step)):
                    if len(dataframe)-i<step:
                        ministep = int((len(dataframe)-i)/8)
                    else:
                        ministep = int(step/8)
                    lines = [dataframe[col][i:i+ministep],dataframe[col][i+ministep: i+ ministep*2],dataframe[col][i+ministep*2: i+ ministep*3],
                             dataframe[col][i+ministep*3: i+ ministep*4],dataframe[col][i+ministep*4: i+ ministep*5],
                             dataframe[col][i+ministep*5: i+ ministep*6],dataframe[col][i+ministep*6: i+ ministep*7],
                             dataframe[col][i+ministep*7: i+ ministep*8]]
                    p = multiprocessing.Pool(processes=8)
                    data = list(p.map(self.convert_line_to_list, lines))
                    p.close()
                    for el in data:
                        datas += el
                    if psutil.virtual_memory().percent >75:
                        self.write_output(output_path + str(col)+ ".csv", datas)
                        del datas, data, lines
                        gc.collect()
                        datas = []
                        written = True
                if written:
                    datas += self.read_output(output_path + str(col)+ ".csv")
            else:
                datas = dataframe[col]
            self.training_data = pd.DataFrame(training_data_dict)
            
    