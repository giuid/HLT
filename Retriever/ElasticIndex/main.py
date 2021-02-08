from ElasticIndex import ElasticIndex 
import regex
import pandas as pd

def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > 0:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)

index= ElasticIndex()
index_name = "gnq_clean"

results = index.search(index_name=index_name, question_text="How tall is LeBron James?", n_results=5)

docs = [doc['_source']['document_text'] for doc in results['hits']['hits']]

pharagraphs = []

for doc in docs:
    for pharagraph in _split_doc(doc):
        pharagraphs.append(pharagraph)
            
lebron = pd.DataFrame({"Paragrafi" : pharagraphs})     
lebron.to_csv("/home/giuseppe/Scrivania/lebron.csv")

    
