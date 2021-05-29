import pandas as pd
import streamlit as st

from KeywordExtraction.inputs import texts
from KeywordExtraction.eval import *
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from load_css import local_css

local_css("style.css")
 



@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=True)
def load_model(name='distilbert-base-nli-mean-tokens'):
    model = SentenceTransformer(name)
    return model
model_list = ['distilbert-base-nli-mean-tokens','paraphrase-MiniLM-L12-v2','LaBSE']


placeholder = st.empty()
text_input = placeholder.text_area("Type in some text you want to analyze", height=300)


sample_text = st.selectbox(
    "Or pick some sample texts", [f"sample {i+1}" for i in range(len(texts))]
)
sample_id = int(sample_text.split(" ")[-1])
text_input = placeholder.text_area(
    "Type in some text you want to analyze", value=texts[sample_id - 1], height=300
)

model_name= st.selectbox(
    "Select a model", [f"{model_list[i]}" for i in range(len(model_list))]
)


model = load_model(model_name)
topk = st.sidebar.slider("Select number of keywords to extract", 5, 20, 10, 1)
min_ngram = st.sidebar.number_input("Min ngram", 1, 5, 1, 1)
max_ngram = st.sidebar.number_input("Max ngram", min_ngram, 5, 3, step=1)
st.sidebar.code(f"ngram_range = ({min_ngram}, {max_ngram})")

try:
    

    params = {
        "doc": text_input,

        "ngram_range": (min_ngram, max_ngram),
        "stop_words": "english",
        'model':model,
        "topk": topk
    }
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text_input)
    wordcloud.to_file("outputs/first_review.png")

    keywords = main(**params)

#     if keywords != []:
#         st.info("Extracted keywords")
#         keywords = pd.DataFrame(keywords, columns=["keyword",'score'])
#         st.table(keywords)
   
    
    
    t = "<div>"
    for i in keywords:
        t+="<span class='highlight blue'>{}</span>".format(i)
    t+='</div>' 
        
#     t = "<div>Hello there my <span class='highlight blue'>name <span class='bold'>yo</span> </span> is <span class='highlight red'>Fanilo <span class='bold'>Name</span></span></div>"

    st.markdown(t, unsafe_allow_html=True) 
    st.image("outputs/first_review.png")
     
except: 
    st.info('Invalid input choices')             