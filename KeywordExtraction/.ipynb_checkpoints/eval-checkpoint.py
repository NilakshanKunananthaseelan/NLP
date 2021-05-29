import numpy as np
import itertools
import warnings
warnings.filterwarnings('ignore')
from .inputs import texts
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def generate_candidates(ngram_range,stop_words,doc):
    count = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names()
    return candidates

def get_embeddings(model,doc,candidates):
    doc_embedding = model.encode([doc])
    candidate_embedding = model.encode(candidates)
    
    return doc_embedding,candidate_embedding

def get_keywords(candidates,doc_embedding,candidate_embedding,topk=5):
    
    distances = cosine_similarity(doc_embedding, candidate_embedding)
    keywords = [candidates[index] for index in distances.argsort()[0][-topk:][::-1]]
#     keywords = [(candidates[index],distances[0][index])for index in distances.argsort()[0][-topk:][::-1]]
    
    return keywords

def main(doc,ngram_range,stop_words,model,topk=5):
    
    candidates = generate_candidates(ngram_range,stop_words,doc)
    
    doc_embed,candidate_embed = get_embeddings(model,doc,candidates)
    
    keywords = get_keywords(candidates,doc_embed,candidate_embed,topk)
     
    return keywords
    

# if __name__=='__main__':
#     doc = texts[-1]
#     stop_words='english'
#     n_gram_range = (1,1)
#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
# #     main(doc,n_gram_range,stop_words,model)
#     params = {
#     "doc": doc,
#     "ngram_range":  n_gram_range,
#     "stop_words": "english",
#     'model':model,"topk":5
# }

# keywords = main(**params)
