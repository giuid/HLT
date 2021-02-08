#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 12:13:44 2020

@author: giuseppe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:43:22 2020

@author: giuseppe
"""

import json
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer 
tk = WhitespaceTokenizer() 
import os
import queue
from itertools import islice
from functools import partial
from itertools import chain
import sqlite3
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
import unicodedata

r = re.compile(r"\s\s+", re.MULTILINE)


# Import gnq articles
articles = os.getcwd() + "/gnq_articles.db"
connection = sqlite3.connect(articles, check_same_thread=False)
cursor = connection.cursor()
cursor.execute("SELECT id, text FROM documents")

wiki_articles ={}
for el in cursor.fetchall():
    wiki_articles[el[0]]=el[1]
cursor.close()


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def read_lines(path,i=0, step=307373):
    
    with open(path) as f:
        lines_gen = islice(f,i,i+step)
        for line in lines_gen:
            yield line
    f.close()
    
def get_contents(line):
    qa = []
    paragraph = json.loads(line)
    if (paragraph['annotations'][0]['long_answer'].get("start_token") !=-1 and paragraph['annotations'][0]['long_answer'].get("end_token") !=-1):
        doc = tk.tokenize(paragraph['document_text'])
        start_long = paragraph['annotations'][0]['long_answer'].get("start_token")
        if (doc[start_long]=="<P>"):
            doc_title = normalize(r.sub(" ", re.sub(r'</?\w+\S*>','', re.findall(r'<H1>.*?</H1>', paragraph['document_text'])[0])).strip())
            doc_text=""
            for el in re.findall(r'<P>.*?</P>', paragraph['document_text']):
                doc_text+=r.sub(" ", re.sub(r'</?\w+\S*>','', el)).strip() + "\n" 
            doc_text=doc_text.strip()
            if(wiki_articles[doc_title]==doc_text):
                if (len(paragraph['annotations'][0]['short_answers'])!=0):
                        if(len(paragraph['annotations'][0]['short_answers'])>1):
                            question=  paragraph['question_text']
                            start_s = paragraph['annotations'][0]['short_answers'][-1]['start_token']
                            end_s = paragraph['annotations'][0]['short_answers'][-1]['end_token']
                            answer = ' '.join(doc[start_s:end_s])
                            title= doc_title
                            qa.append((title, question, answer))        
                            for i in range(len(paragraph['annotations'][0]['short_answers'])-1):
                                question=  paragraph['question_text']
                                start_s = paragraph['annotations'][0]['short_answers'][i]['start_token']
                                end_s = paragraph['annotations'][0]['short_answers'][i]['end_token']
                                answer = ' '.join(doc[start_s:end_s])
                                title = doc_title + " (" + str(i) + ") "
                                qa.append((title, question, answer))
                        else:
                            title = doc_title
                            question =  paragraph['question_text']
                            start_s = paragraph['annotations'][0]['short_answers'][0]['start_token']
                            end_s = paragraph['annotations'][0]['short_answers'][0]['end_token']
                            answer = ' '.join(doc[start_s:end_s])
                            qa.append((title, question, answer))
                    
    return qa


# Import gnq 
gnq = "/media/giuseppe/DATA/LINUX/simplified-nq-train.jsonl"

conn = sqlite3.connect('gnq_qa.db')
c = conn.cursor()
c.execute("CREATE TABLE QA ( id NOT NULL, question, answer);")
lines = 307373
count=0
workers = ProcessPool(processes = 4 )


with tqdm(total=lines) as pbar:
    for pair in tqdm(workers.imap(get_contents, read_lines(gnq))):
        count += len(pair)
        c.executemany('''INSERT INTO QA VALUES (?,?,?)''', pair)
        pbar.update()
print('Read %d docs.' % count)
print('Committing...')
conn.commit()
conn.close()


