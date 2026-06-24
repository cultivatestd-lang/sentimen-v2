import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Sentimen MBG", layout="wide")
st.title("🍽️ Analisis Sentimen Program Makan Bergizi Gratis")

# ─── CACHE ─────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    import nltk
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    factory_stop = StopWordRemoverFactory()
    stopword_list = factory_stop.get_stop_words()
    return stemmer, set(stopword_list)

@st.cache_data
def load_data():
    df = pd.read_csv('data_mbg_gabungan.csv')
    df = df[['teks_tweet', 'created_at']].copy()
    df.columns = ['teks', 'tanggal']
    df.dropna(subset=['teks'], inplace=True)
    df = df[~df['teks'].str.startswith('RT @', na=False)]
    df.drop_duplicates(subset=['teks'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def load_label_cache():
    if os.path.exists('data_berlabel.csv'):
        return pd.read_csv('data_berlabel.csv')
    return None

# ─── DATA ──────────────────────────────────────────────────────
stemmer, stopword_set = load_nlp()
df = load_data()
st.success(f"✅ {len(df)} tweets unik")

# ─── KAMUS ─────────────────────────────────────────────────────
kamus_slang = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'tak': 'tidak', 'tdk': 'tidak', 'gk': 'tidak', 'g': 'tidak',
    'yg': 'yang', 'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena',
    'utk': 'untuk', 'tuk': 'untuk', 'buat': 'untuk',
    'udah': 'sudah', 'udh': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'blum': 'belum', 'lg': 'lagi', 'lgi': 'lagi',
    'jg': 'juga', 'aja': 'saja', 'aj': 'saja',
    'emg': 'memang', 'emang': 'memang', 'bgt': 'banget', 'bngt': 'banget',
    'klo': 'kalau', 'klu': 'kalau', 'kl': 'kalau',
    'sy': 'saya', 'gue': 'saya', 'gw': 'saya', 'aku': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'hrs': 'harus',
    'bs': 'bisa', 'bsa': 'bisa', 'dr': 'dari', 'pd': 'pada',
    'tp': 'tapi', 'tpi': 'tapi', 'mau': 'ingin', 'mo': 'ingin',
    'mbg': 'makan bergizi gratis', 'makber': 'makan bergizi',
    'pmb': 'program makan bergizi', 'prabowo': 'presiden',
}

kamus_stem_kustom = {
    'keburukan': 'buruk', 'keburu': 'buruk', 'ketidakadilan': 'tidak adil',
    'kebodohan': 'bodoh', 'kerusakan': 'rusak', 'kebencian': 'benci',
    'kejahatan': 'jahat', 'kegagalan': 'gagal', 'kebahagiaan': 'bahagia',
    'keberhasilan': 'berhasil', 'kesehatan': 'sehat', 'kesuksesan': 'sukses',
    'kesedihan': 'sedih', 'kekecewaan': 'kecewa', 'kemarahan': 'marah',
    'kemiskinan': 'miskin', 'kekerasan': 'keras', 'kerugian': 'rugi',
    'keracunan': 'racun', 'meracuni': 'racun', 'mencurangi': 'curang',
}

lexicon_positif = {
    'baik', 'bagus', 'mantap', 'keren', 'hebat', 'senang', 'bahagia',
    'puas', 'bangga', 'bermanfaat', 'sukses', 'berhasil', 'enak', 'nikmat',
    'sehat', 'aman', 'nyaman', 'adil', 'membantu', 'tepat', 'layak', 'pantas',
    'terbaik', 'semangat', 'positif', 'syukur', 'bersyukur',
    'kebahagiaan', 'kesuksesan', 'keberhasilan', 'kesehatan', 'keadilan',
    'terbantu',
}

lexicon_negatif = {
    'buruk', 'jelek', 'parah', 'busuk', 'rusak', 'sakit', 'sedih',
    'kecewa', 'kesal', 'marah', 'benci', 'gagal', 'rugi', 'curang',
    'penipuan', 'bohong', 'jahat', 'bahaya', 'takut', 'hancur',
    'racun', 'keracunan', 'sampah', 'kacau', 'korupsi',
    'keburukan', 'kebodohan', 'kemiskinan', 'kebencian', 'kesedihan',
    'kekecewaan', 'ketidakadilan', 'kerugian', 'kerusakan', 'kegagalan',
    'mengecewakan', 'menyedihkan', 'memalukan', 'merugikan',
    'terburuk', 'lapar', 'mual', 'pusing', 'diare',
}

custom_stopwords = {
    'mbg', 'makan', 'bergizi', 'gratis', 'program', 'pemerintah',
    'indonesia', 'anak', 'sekolah', 'siswa', 'saja', 'sudah',
    'juga', 'ini', 'itu', 'ada', 'dengan', 'yang', 'dan', 'di',
    'ke', 'dari', 'untuk', 'pada', 'dalam', 'akan', 'tidak', 'bisa'
}

# ─── CLEANING ──────────────────────────────────────────────────
def cleaning_teks(teks):
    if pd.isna(teks): return ''
    teks = str(teks)
    emoji = re.compile('[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
                       '\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
                       '\u2600-\u27BF\uFE00-\uFE0F]', re.UNICODE)
    teks = emoji.sub('', teks)
    teks = re.sub(r'http\S+|www\.\S+', '', teks)
    teks = re.sub(r'@\w+', '', teks)
    teks = re.sub(r'#(\w+)', r'\1', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = teks.lower()
    teks = re.sub(r'\s+', ' ', teks).strip()
    teks = re.sub(r'(.)\1{2,}', r'\1', teks)
    kata_list = teks.split()
    return ' '.join([kamus_slang.get(k, k) for k in kata_list])

def tokenize_dan_clean(teks):
    if not teks or teks.strip() == '': return ''
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(teks)
    semua_stopword = stopword_set | custom_stopwords
    tokens = [t for t in tokens if t not in semua_stopword and len(t) > 2]
    tokens = [kamus_stem_kustom.get(t, stemmer.stem(t)) for t in tokens]
    return ' '.join(tokens)

def fitur_sentimen(teks_series):
    hasil = []
    for teks in teks_series:
        if not isinstance(teks, str) or teks.strip() == '':
            hasil.append([0, 0, 0, 0.0, 0.0, 0.0, 0.5])
            continue
        kata = teks.lower().split()
        p = sum(1 for k in kata if k in lexicon_positif)
        n = sum(1 for k in kata if k in lexicon_negatif)
        total_k = p + n
        hasil.append([
            p, n, p - n,
            p / total_k if total_k > 0 else 0.0,
            n / total_k if total_k > 0 else 0.0,
            round(((p - n) / (total_k + 1)) * 100, 2),
            round((((p - n) / (total_k + 1)) * 100 / 100 + 1) / 2 if total_k > 0 else 0.5, 4)
        ])
    return np.array(hasil)

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Kontrol")
    use_cache = st.checkbox("Pakai cache label (lebih cepat)", True)
    fast_mode = st.checkbox("Mode cepat (4 model, skip CV)", True)

# ─── PROSES ────────────────────────────────────────────────────
if st.button("🚀 Jalankan Analisis"):
    bar = st.progress(0, "Membersihkan teks...")
    df['teks_clean'] = df['teks'].apply(cleaning_teks)
    df['teks_proses'] = df['teks_clean'].apply(tokenize_dan_clean)
    df = df[df['teks_proses'].str.strip() != ''].reset_index(drop=True)

    bar.progress(20, "Melabel sentimen...")

    if use_cache and os.path.exists('data_berlabel.csv'):
        cache = pd.read_csv('data_berlabel.csv')
        df['sentimen'] = cache['sentimen']
        st.info("📦 Label dari cache (data_berlabel.csv)")
    else:
        from transformers import pipeline
        from tqdm import tqdm
        pipe = pipeline("text-classification",
                        model="w11wo/indonesian-roberta-base-sentiment-classifier",
                        tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
                        device=-1)
        teks_list = df['teks_clean'].fillna('').tolist()
        labels = []
        for i in range(0, len(teks_list), 32):
            batch = [t[:512] if t.strip() else 'netral' for t in teks_list[i:i+32]]
            try:
                for h in pipe(batch):
                    l = h['label'].lower()
                    labels.append('Positif' if l == 'positive' else 'Negatif' if l == 'negative' else 'Netral')
            except:
                labels.extend(['Netral'] * len(batch))
            bar.progress(20 + int(i / len(teks_list) * 30), f"Label {i}/{len(teks_list)}")
        df['sentimen'] = labels
        df[['teks', 'teks_clean', 'teks_proses', 'tanggal', 'sentimen']].to_csv('data_berlabel.csv', index=False)
        st.success("💾 Label disimpan ke data_berlabel.csv")

    # ─── DISTRIBUSI ──────────────────────────────────────────
    bar.progress(55, "Visualisasi distribusi...")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribusi Sentimen")
        counts = df['sentimen'].value_counts()
        total = len(df)
        for s in ['Positif', 'Negatif', 'Netral']:
            jml = counts.get(s, 0)
            pct = jml / total * 100
            st.metric(s, f"{jml} ({pct:.1f}%)")
            st.progress(pct / 100)

    with col2:
        fig, ax = plt.subplots()
        colors = {'Positif': '#2ecc71', 'Negatif': '#e74c3c', 'Netral': '#95a5a6'}
        c = [colors.get(s, 'gray') for s in counts.index]
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=c, startangle=90)
        ax.set_title("Distribusi Sentimen")
        st.pyplot(fig)

    # ─── RATA-RATA LEXICON PER KELAS ────────────────────────
    st.subheader("📊 Rata-rata Skor Lexicon per Kelas")
    data_lex = []
    for s in ['Positif', 'Negatif', 'Netral']:
        tg = ' '.join(df[df['sentimen'] == s]['teks_clean'].dropna())
        if tg.strip():
            kt = tg.lower().split()
            p = sum(1 for k in kt if k in lexicon_positif)
            n = sum(1 for k in kt if k in lexicon_negatif)
            sk = ((p - n) / (p + n + 1)) * 100
            data_lex.append({'Kelas': s, 'Lexicon Polarity': f"{sk:+.1f}%", 'Pos': p, 'Neg': n})
    st.dataframe(pd.DataFrame(data_lex), use_container_width=True)

    # ─── TF-IDF + FITUR SENTIMEN ────────────────────────────
    bar.progress(65, "Ekstraksi fitur...")
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                             min_df=2, max_df=0.95, sublinear_tf=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df['teks_clean'], df['sentimen'], test_size=0.2, random_state=42, stratify=df['sentimen']
    )
    X_tr_tf = tfidf.fit_transform(X_train)
    X_te_tf = tfidf.transform(X_test)
    fs_tr = fitur_sentimen(X_train)
    fs_te = fitur_sentimen(X_test)
    X_tr = hstack([X_tr_tf, csr_matrix(fs_tr)])
    X_te = hstack([X_te_tf, csr_matrix(fs_te)])

    # ─── TRAIN MODEL ─────────────────────────────────────────
    bar.progress(75, "Melatih model...")
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    models = [
        ('SVM', SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
    ]
    if not fast_mode:
        models.append(('Naive Bayes', MultinomialNB(alpha=1.0)))

    hasil = []
    for nama, model in models:
        if nama == 'Naive Bayes':
            scaler = MinMaxScaler()
            X_tr_m = csr_matrix(scaler.fit_transform(X_tr.toarray()))
            X_te_m = csr_matrix(scaler.transform(X_te.toarray()))
            model.fit(X_tr_m, y_train)
            yp = model.predict(X_te_m)
        else:
            model.fit(X_tr, y_train)
            yp = model.predict(X_te)
        hasil.append({
            'Model': nama,
            'Accuracy': round(accuracy_score(y_test, yp), 4),
            'Precision': round(precision_score(y_test, yp, average='weighted', zero_division=0), 4),
            'Recall': round(recall_score(y_test, yp, average='weighted', zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, yp, average='weighted', zero_division=0), 4),
        })

    bar.progress(90, "Selesai...")
    st.subheader("🏆 Perbandingan Model")
    st.dataframe(pd.DataFrame(hasil).set_index('Model'), use_container_width=True)
    best = max(hasil, key=lambda x: x['F1-Score'])
    st.success(f"🏆 Model terbaik: **{best['Model']}** (F1: {best['F1-Score']})")

    # ─── CONTOH PREDIKSI ────────────────────────────────────
    st.subheader("🔍 Uji Prediksi")
    teks_input = st.text_input("Masukkan tweet:", "banyak anak keracunan makanan di sekolah")
    if teks_input:
        tc = cleaning_teks(teks_input)
        tv = tfidf.transform([tc])
        ts = fitur_sentimen([tc])
        tg = hstack([tv, csr_matrix(ts)])
        # Pakai model SVM
        svm = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
        svm.fit(X_tr, y_train)
        pred = svm.predict(tg)[0]
        kata = teks_input.lower().split()
        p = sum(1 for k in kata if k in lexicon_positif)
        n = sum(1 for k in kata if k in lexicon_negatif)
        skor = ((p - n) / (p + n + 1)) * 100
        st.metric("Prediksi", pred)
        st.metric("Skor Lexicon", f"{skor:+.1f}%")

    bar.progress(100, "✅ Selesai!")
else:
    st.info("👈 Klik **Jalankan Analisis** untuk memulai")
    st.caption("Dataset: data_mbg_gabungan.csv")
