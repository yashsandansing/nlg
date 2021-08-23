#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# In[3]:


model = load_model("model.h5")
pickle_in = open('tokenizer.pkl', 'rb') 
tokenizer = pickle.load(pickle_in)


# In[6]:

def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return " ".join(tokens)

# generate a sequence from a language model
def generate_seq(seq_length, seed_text, n_words):
    result = list()
    in_text = clean_doc(seed_text)
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)


# In[7]:


def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:gray;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Natural Language Generation ML App</h1> 
    </div> 
    <p>INFO: This is an application that uses an LSTM network to generate natural language text.</p>
    <p>This model is trained on scientific data including topics like 'deep learning', 'covid-19', 'human connectome', 'virtual-reality', 'brain machine interfaces', 'electroactive polymers', 'pedot electrodes', and 'neuroprosthetics'.
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    sentence = st.text_input('Input your sentence here:') 

    if st.button("Predict"): 
        result = generate_seq(50, sentence, 50)
        st.success('{}'.format(sentence))
        st.success('{}'.format(result))


# In[8]:


if __name__=='__main__': 
    main()




