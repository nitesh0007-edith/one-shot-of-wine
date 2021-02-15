import streamlit as st
import tensorflow
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import tensorflow
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model and other files

loaded_model = tensorflow.keras.models.load_model('model.h5')
dataset=pd.read_csv('final_data.csv', names= ["aaa","description", "target"])
dataset=dataset.drop(["aaa"],axis=1)
df = dataset.iloc[1:150000]
labels = df.target.values

def check(num1,num2):
    if num1 == num2:
        return 'Similar'
    else:
        return 'Dissimilar'

def main ():
    html_temp = """
        <div style="padding:10px; background-color:#722f37; ">
            <h2 style="color:white;text-align:center;"> ONE-SHOT OF WINE </h2>
        </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("""
        Given two sentences, the app predicts whether
        they fall into the same labelled category—whether they’re about the same thing, written by the same
        person, or have the same style or meaning.
    """)

    sentence1 = "@"
    sentence2 = "@"

    sentence1 += st.text_input('Input your first sentence here:')
    sentence2 += st.text_input('Input your second sentence here:')

    inputs = [sentence1, sentence2]

    out = []
    for i in inputs:
        review = i
        lower = map(str.lower, review)
        tok = list(map(word_tokenize, lower))
        stop_words = stopwords.words("english")
        review_text = [str for str in tok if str not in stop_words]
        ps = PorterStemmer()
        output = []
        for sentence in review_text:
            output.append(" ".join([ps.stem(i) for i in sentence]))

        MAX_NB_WORDS = 50000
        MAX_SEQUENCE_LENGTH = 250

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        seq = tokenizer.texts_to_sequences(output)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

        pred = loaded_model.predict(padded)
        a = labels[np.argmax(pred)]
        out.append(a)

    if st.button('Check the Similarity'):
        st.success(check(out[0],out[1]))

if __name__ == '__main__':
    main()


