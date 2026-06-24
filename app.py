import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

st.set_page_config(page_title="Analisis Sentimen MBG", page_icon="🍽️", layout="centered")

st.title("🍽️ Aplikasi Analisis Sentimen Program Makan Bergizi Gratis")
st.write("Aplikasi ini memprediksi sentimen masyarakat menggunakan model Support Vector Machine (SVM).")

@st.cache_resource
def load_model():
    with open('svm_tuned_mbg.pkl', 'rb') as f:
        loaded_object = pickle.load(f)
    
    # Ambil estimator terbaik jika dibungkus GridSearchCV/RandomizedSearchCV
    if hasattr(loaded_object, 'best_estimator_'):
        return loaded_object.best_estimator_
    return loaded_object

try:
    model_pipeline = load_model()
    
    user_tweet = st.text_area("Masukkan teks tweet/komentar di sini:", placeholder="Tulis pendapat di sini...")

    if st.button("Analisis Sentimen", type="primary"):
        if user_tweet.strip() != "":
            cleaned_tweet = re.sub(r'http\S+|www\.\S+', '', user_tweet)
            cleaned_tweet = re.sub(r'@\w+', '', cleaned_tweet)
            cleaned_tweet = cleaned_tweet.lower()
            
            # Kembali menggunakan predict() standar
            prediction = model_pipeline.predict([cleaned_tweet])[0]
            
            st.subheader("Hasil Analisis:")
            if prediction == 'Positif':
                st.success(f"Sentimen: **{prediction}** 😊🟢")
            elif prediction == 'Negatif':
                st.error(f"Sentimen: **{prediction}** 😡🔴")
            else:
                st.warning(f"Sentimen: **{prediction}** 😐🟡")
        else:
            st.error("Silakan masukkan teks terlebih dahulu!")

except Exception as e:
    st.error(f"Error memuat model: {e}")
