#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:50:11 2020

@author: giuseppe
"""
import sqlite3
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
import json
import re
import os
from itertools import islice
import unicodedata
import nltk
from nltk.tokenize import WhitespaceTokenizer 
tk = WhitespaceTokenizer() 

r = re.compile(r"\s\s+", re.MULTILINE)
 
def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

#307373
def read_lines(path,i=0, step=307373):
    
    with open(path) as f:
        lines_gen = islice(f,i,i+step)
        for line in lines_gen:
            #l = json.loads(line)
            yield line
    f.close()



def get_contents(line):
    documents = [("empty_title", "empty_text")]
    paragraph = json.loads(line)
    doc_title = normalize(r.sub(" ", re.sub(r'</?\w+\S*>','', re.findall(r'<H1>.*?</H1>', paragraph['document_text'])[0])).strip())
    doc_text=""
    for el in re.findall(r'<P>.*?</P>', paragraph['document_text']):
        doc_text+=r.sub(" ", re.sub(r'</?\w+\S*>','', el)).strip() + "\n" 
    doc_text=doc_text.strip()
    documents.append((doc_title, doc_text))
    
                    
    return documents 

#####################################################################################

gnq = "/media/giuseppe/DATA/LINUX/simplified-nq-train.jsonl"


conn = sqlite3.connect('gnq_articles.db')
c = conn.cursor()
c.execute("CREATE TABLE documents ( id PRIMARY KEY, text CHECK (id != 'empty_title' AND text != 'empty_text'));")
lines = 307373
count=0
workers = ProcessPool(processes = 4)

with tqdm(total=lines) as pbar:
    for pair in tqdm(workers.imap(get_contents, read_lines(gnq))):
        count += len(pair)
        c.executemany('''INSERT OR IGNORE INTO documents VALUES (?,?)''', pair)
        pbar.update()
print('Read %d docs.' % count)
print('Committing...')
conn.commit()
conn.close()
