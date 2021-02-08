#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:17:41 2020

@author: omen
"""

import os
import json
import spacy
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
import gensim.downloader as api
import warnings
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers import pipeline, AutoTokenizer


# initialize models
nlp = spacy.load("en_core_web_sm")
word_vectors = api.load("glove-wiki-gigaword-50")
unmasker = pipeline('fill-mask', model="bert-base-uncased", tokenizer="bert-base-uncased",framework='pt', device=0)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

class QueryExpander:
    '''
    Query expansion utility that augments ElasticSearch queries with optional techniques
    including Named Entity Recognition and Synonym Expansion
    
    Args:
        question_text
        entity (dict) - Ex. {'spacy_model': nlp}
        synonym (dict) - Ex. {'gensim_model': word_vectors, 'n_syns': 3} OR
                                  {'MLM': unmasker, 'tokenizer': base_tokenizer, 'n_syns': 3, 'threshold':0.3}
    
    '''

    
    
    def __init__(self, question_text, entity=False, synonym=None, n_syns = 2):
        
        
        self.question_text = question_text
        
        if (entity):
            
            self.entity = {'spacy_model': nlp}
        else:
            self.entity = entity
        
       
        if (synonym == "static"):
            self.synonym = {'gensim_model': word_vectors,
                'n_syns': n_syns}
        elif (synonym == "contextual"):
            self.synonym = {'MLM': unmasker, 
                'tokenizer': tokenizer, 
                'n_syns': n_syns,
                'threshold':0}
        else:
            self.synonym = None

        if self.synonym and not self.entity:
            raise Exception('Cannot do synonym expansion without NER! Expanding synonyms\
                            on named entities reduces recall.')

        if self.synonym or self.entity:
            self.nlp = self.entity['spacy_model']
            self.doc = self.nlp(self.question_text)
        
        self.build_query()
        
    def build_query(self):

        # build entity subquery
        if self.entity:
            self.extract_entities()
        
        # identify terms to expand
        if self.synonym:
            self.identify_terms_to_expand()
        
        # build question subquery
        self.construct_question_query()
        
        # combine subqueries
        sub_queries = []
        sub_queries.append(self.question_sub_query)
        if hasattr(self, 'entity_sub_queries'):
            sub_queries.extend(self.entity_sub_queries)
            
        query = Q('bool', should=[*sub_queries])
        self.query = query
        
    
    def extract_entities(self):
        '''
        Extracts named entities using spaCy and constructs phrase match subqueries
        for each entity. Saves both entities and subqueries as attributes.
        
        '''
        
        entity_list = [entity.text.lower() for entity in self.doc.ents]
        entity_sub_queries = []
        
        for ent in entity_list:
            eq = Q('multi_match',
                   query=ent,
                   type='phrase',
                   fields=['document_title', 'document_text'])
            
            entity_sub_queries.append(eq)
        
        self.entities = entity_list
        self.entity_sub_queries = entity_sub_queries
        
        
    def identify_terms_to_expand(self):
        '''
        Identify terms in the question that are eligible for expansion
        per a set of defined rules
        
        '''
        if hasattr(self, 'entities'):
            # get unique list of entity tokens
            entity_terms = [ent.split(' ') for ent in self.entities]
            entity_terms = [ent for sublist in entity_terms for ent in sublist]
        else:
            entity_terms = []
    
        # terms to expand are not part of entity, a stopword, numeric, etc.
        entity_pos = ["NOUN","VERB","ADJ","ADV"]
        terms_to_expand = [idx_term for idx_term in enumerate(self.doc) if \
                           (idx_term[1].lower_ not in entity_terms) and (not idx_term[1].is_stop)\
                            and (not idx_term[1].is_digit) and (not idx_term[1].is_punct) and 
                            (not len(idx_term[1].lower_)==1 and idx_term[1].is_alpha) and
                            (idx_term[1].pos_ in entity_pos)]
        
        self.terms_to_expand = terms_to_expand

        
    def construct_question_query(self):
        '''
        Builds a multi-match query from the raw question text extended with synonyms 
        for any eligible terms

        '''
        
        if hasattr(self, 'terms_to_expand'):
            
            syns = []
            for i, term in self.terms_to_expand:

                if 'gensim_model' in self.synonym.keys():
                    syns.extend(self.gather_synonyms_static(term))

                elif 'MLM' in self.synonym.keys():
                    syns.extend(self.gather_synonyms_contextual(i, term))

            syns = list(set(syns))
            syns = [syn for syn in syns if (syn.isalpha() and self.nlp(syn)[0].pos_ != 'PROPN')]
            
            question = self.question_text + ' ' + ' '.join(syns)
            self.expanded_question = question
            self.all_syns = syns
        
        else:
            question = self.question_text
        
        qq = Q('multi_match',
               query=question,
               type='most_fields',
               fields=['document_title', 'document_text'])
        
        self.question_sub_query = qq


    def gather_synonyms_contextual(self, token_index, token):
        '''
        Takes in a token, and returns specified number of synonyms as defined by
        predictions from a masked language model
        
        '''
        
        tokens = [token.text for token in self.doc]
        tokens[token_index] = self.synonym['tokenizer'].mask_token
        
        terms = self.predict_mask(text = ' '.join(tokens), 
                                    unmasker = self.synonym['MLM'],
                                    tokenizer = self.synonym['tokenizer'],
                                    threshold = self.synonym['threshold'],
                                    top_n = self.synonym['n_syns'])
        
        return terms

    @staticmethod
    def predict_mask(text, unmasker, tokenizer, threshold=0, top_n=2):
        '''
        Given a sentence with a [MASK] token in it, this function will return the most 
        contextually similar terms to fill in the [MASK]
        
        '''

        preds = unmasker(text)
        tokens = [tokenizer.convert_ids_to_tokens(pred['token']) for pred in preds if pred['score'] > threshold]
        
        return tokens[:top_n]
        

    def gather_synonyms_static(self, token):
        '''
        Takes in a token and returns a specified number of synonyms defined by
        cosine similarity of word vectors. Uses stemming to ensure none of the
        returned synonyms share the same stem (ex. photo and photos can't happen)
        
        '''
        try:
            syns = self.synonym['gensim_model'].similar_by_word(token.lower_)

            lemmas = []
            final_terms = []
            for item in syns:
                term = item[0]
                lemma = self.nlp(term)[0].lemma_

                if lemma in lemmas:
                    continue
                else:
                    lemmas.append(lemma)
                    final_terms.append(term)
                    if len(final_terms) == self.synonym['n_syns']:
                        break
        except:
            final_terms = []

        return final_terms

    def explain_expansion(self, entities=True):
        '''
        Print out an explanation for the query expansion methodology
        
        '''
        
        print('Question:', self.question_text, '\n')
        
        if entities:
            print('Found Entities:', self.entities, '\n')
        
        if hasattr(self, 'terms_to_expand'):
            
            print('Synonym Expansions:')
        
            for i, term in self.terms_to_expand:
                
                if 'gensim_model' in self.synonym.keys():
                    print(term, '-->', self.gather_synonyms_static(term))
                
                elif 'MLM' in self.synonym.keys():
                    print(term, '-->', self.gather_synonyms_contextual(i,term))
            
                else:
                    print('Question text has no terms to expand.')
                    
            print()
            print('Expanded Question:', self.expanded_question, '\n')
        
        print('Elasticsearch Query:\n', self.query)
        
    
    def get_query(self):
        return self.query
        
    def search(self, client, index_name):
        s = Search(using=client, index=index_name)
        s = s.query(self.query)
        response = s.execute()
        return response
    
    

    

'''


question = "how many rose species are found in the Montreal Botanical Garden?"
index_name='gnq_clean'
config = {'host':'localhost', 'port':9200}
client = Elasticsearch([config])
client.ping()    

#1) Normal
normal = QueryExpander(question)
response = normal.search(client, index_name=index_name)
a = response.to_dict()
normal.get_query()

#2) Entity Named 
qe_ner = QueryExpander(question, entity = True)
response = qe_ner.search(client, index_name=index_name)
b = response.to_dict()
qe_ner.get_query()

#3) Static Embeddings
qe_static = QueryExpander(question, entity = True, synonym = 'static')
response = qe_static.search(client, index_name=index_name)
c = response.to_dict()
qe_static.get_query()

#4) Contextual Embeddings
qe_contextual = QueryExpander(question, entity = True, synonym = 'contextual', n_syns=2)
response = qe_contextual.search(client, index_name=index_name)
d = response.to_dict()
print(qe_contextual.get_query())

'''

