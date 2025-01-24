import streamlit as st
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

st.title('IMDB Movie Review')
st.write('Enter a movie review to classify it as positive or negative.')
user_input=st.text_area('Movie Review')


word_index=imdb.get_word_index()

# load the model
model=load_model('imdb_movie_review_model.h5')

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=pad_sequences([encoded_review],maxlen=500)
    return padded_review


if st.button('Classify'):
    preprocessed_text=preprocess_text(text=user_input)
    prediction=model.predict(preprocessed_text)
    sentiment= 'Positive' if prediction[0][0]>0.5 else 'Negative'
    score=prediction[0][0]

    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction score:{score:.2f}')





