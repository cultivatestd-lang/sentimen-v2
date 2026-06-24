# -*- coding: utf-8 -*-
"""Analisis Sentimen Program Makan Bergizi Gratis — SVM + Komparasi Model"""

# ═══════════════════════════════════════════════════════════════
# KONFIGURASI CEPAT (ubah sesuai kebutuhan Streamlit / Colab)
# ═══════════════════════════════════════════════════════════════
SKIP_LABEL_INDOBERT  = True   # True = pakai cache data_berlabel.csv (lebih cepat)
SKIP_OPTUNA          = True   # True = lewati hyperparameter tuning Optuna
SKIP_MODEL_LAMBAN    = True   # True = hanya SVM/LR/RF/NB, skip KNN/DT/GB
CV_FOLDS             = 3      # lebih kecil = lebih cepat (default 5)

# ═══════════════════════════════════════════════════════════════
# 1. IMPORT LIBRARY
# ═══════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, re, warnings, time, os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
warnings.filterwarnings('ignore')

# NLP
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

print('✅ Library siap')

# ═══════════════════════════════════════════════════════════════
# 2. LOAD DATA
# ═══════════════════════════════════════════════════════════════
print('\n📂 Loading data...')
df_raw = pd.read_csv('data_mbg_gabungan.csv')
KOLOM_TEKS = 'teks_tweet'
KOLOM_TANGGAL = 'created_at'

df = df_raw[[KOLOM_TEKS, KOLOM_TANGGAL]].copy()
df.columns = ['teks', 'tanggal']
df.dropna(subset=['teks'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Hapus retweet & duplikat
df = df[~df['teks'].str.startswith('RT @', na=False)]
df.drop_duplicates(subset=['teks'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f'✅ Data: {len(df)} tweets unik')

# ═══════════════════════════════════════════════════════════════
# 3. KAMUS SLANG & SENTIMEN
# ═══════════════════════════════════════════════════════════════
kamus_slang = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'tak': 'tidak', 'tdk': 'tidak', 'gk': 'tidak', 'g': 'tidak',
    'yg': 'yang', 'dgn': 'dengan', 'krn': 'karena', 'karna': 'karena',
    'utk': 'untuk', 'tuk': 'untuk', 'buat': 'untuk',
    'udah': 'sudah', 'udh': 'sudah', 'sdh': 'sudah',
    'blm': 'belum', 'blum': 'belum', 'lg': 'lagi', 'lgi': 'lagi',
    'jg': 'juga', 'aja': 'saja', 'aj': 'saja',
    'emg': 'memang', 'emang': 'memang',
    'bgt': 'banget', 'bngt': 'banget',
    'klo': 'kalau', 'klu': 'kalau', 'kl': 'kalau',
    'sy': 'saya', 'gue': 'saya', 'gw': 'saya', 'aku': 'saya',
    'lo': 'kamu', 'lu': 'kamu', 'hrs': 'harus',
    'bs': 'bisa', 'bsa': 'bisa', 'dr': 'dari', 'pd': 'pada',
    'tp': 'tapi', 'tpi': 'tapi', 'mau': 'ingin', 'mo': 'ingin',
    'mbg': 'makan bergizi gratis', 'makber': 'makan bergizi',
    'pmb': 'program makan bergizi', 'prabowo': 'presiden',
}

kamus_stem_kustom = {
    'keburukan': 'buruk', 'keburu': 'buruk',
    'ketidakadilan': 'tidak adil', 'keterpurukan': 'terpuruk',
    'kebodohan': 'bodoh', 'kerusakan': 'rusak', 'kerusuhan': 'rusuh',
    'kebencian': 'benci', 'kejahatan': 'jahat', 'kegagalan': 'gagal',
    'kebahagiaan': 'bahagia', 'keberhasilan': 'berhasil',
    'kesehatan': 'sehat', 'kebersihan': 'bersih', 'keamanan': 'aman',
    'keindahan': 'indah', 'kenyamanan': 'nyaman', 'kemiskinan': 'miskin',
    'kekerasan': 'keras', 'keterlambatan': 'lambat', 'kerugian': 'rugi',
    'kesuksesan': 'sukses', 'kesedihan': 'sedih', 'kekecewaan': 'kecewa',
    'kemarahan': 'marah', 'memperparah': 'parah',
    'meracuni': 'racun', 'mencurangi': 'curang',
}

lexicon_positif = set([
    'baik', 'bagus', 'mantap', 'keren', 'hebat', 'indah', 'senang',
    'bahagia', 'gembira', 'puas', 'bangga', 'suka', 'cinta', 'sayang',
    'peduli', 'bermanfaat', 'berguna', 'tepat', 'cerdas', 'pintar',
    'berhasil', 'sukses', 'lancar', 'bersih', 'sehat', 'kuat',
    'berkualitas', 'murah', 'terjangkau', 'memadai', 'cukup', 'enak',
    'lezat', 'nikmat', 'segar', 'nyaman', 'aman', 'damai', 'rukun',
    'membantu', 'memudahkan', 'memperbaiki', 'meningkatkan',
    'tanggap', 'cepat', 'sigap', 'transparan', 'adil', 'merata',
    'efektif', 'efisien', 'produktif', 'layak', 'pantas',
    'terbaik', 'terhebat', 'terkeren', 'terbesar',
    'semangat', 'antusias', 'optimis', 'positif',
    'syukur', 'bersyukur', 'berterima kasih',
    'kebahagiaan', 'kesuksesan', 'keberhasilan', 'kebaikan',
    'kesehatan', 'keindahan', 'kenyamanan', 'keamanan', 'keadilan',
    'terbantu', 'terbantukan', 'terayomi',
])

lexicon_negatif = set([
    'buruk', 'jelek', 'parah', 'busuk', 'rusak', 'sakit', 'sedih',
    'kecewa', 'kesal', 'marah', 'benci', 'gagal', 'rugi', 'celaka',
    'susah', 'sulit', 'curang', 'penipuan', 'bohong', 'jahat',
    'bahaya', 'berbahaya', 'ancam', 'ancaman',
    'takut', 'cemas', 'khawatir', 'gelisah',
    'menderita', 'tersiksa', 'hancur', 'runtuh',
    'racun', 'keracunan', 'beracun',
    'sampah', 'sia-sia', 'percuma', 'mubazir', 'ngawur',
    'amburadul', 'berantakan', 'kacau',
    'keburukan', 'kebodohan', 'kemiskinan', 'kekerasan',
    'kebencian', 'kemarahan', 'kesedihan', 'kekecewaan',
    'ketidakadilan', 'ketidakjujuran',
    'kerugian', 'kerusakan', 'kegagalan',
    'mengecewakan', 'menyedihkan', 'mengerikan',
    'memprihatinkan', 'memalukan', 'merugikan', 'membahayakan',
    'memperburuk', 'memperparah',
    'terburuk', 'terparah', 'terjelek',
    'korupsi', 'kriminal', 'payah', 'loyo',
    'mangkrak', 'terbengkalai', 'telantar',
    'lapar', 'begah', 'mual', 'pusing', 'alergi', 'kembung', 'diare',
])

# ═══════════════════════════════════════════════════════════════
# 4. CLEANING TEKS
# ═══════════════════════════════════════════════════════════════
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def stem_kustom(kata):
    if kata in kamus_stem_kustom:
        return kamus_stem_kustom[kata]
    return stemmer.stem(kata)

def cleaning_teks(teks):
    if pd.isna(teks):
        return ''
    teks = str(teks)
    # Hapus emoji (range unicode emoji)
    emoji_pattern = re.compile(
        '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        '\U0001F1E0-\U0001F1FF\u2600-\u27BF\uFE00-\uFE0F]', re.UNICODE)
    teks = emoji_pattern.sub('', teks)
    teks = re.sub(r'http\S+|www\.\S+', '', teks)
    teks = re.sub(r'@\w+', '', teks)
    teks = re.sub(r'#(\w+)', r'\1', teks)
    teks = re.sub(r'\d+', '', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = teks.lower()
    teks = re.sub(r'\s+', ' ', teks).strip()
    # Normalisasi huruf berulang: "bagusssss" → "bagus"
    teks = re.sub(r'(.)\1{2,}', r'\1', teks)
    # Normalisasi slang
    kata_list = teks.split()
    teks = ' '.join([kamus_slang.get(k, k) for k in kata_list])
    return teks

# Stopword
factory_stop = StopWordRemoverFactory()
stopword_list = factory_stop.get_stop_words()
custom_stopwords = [
    'mbg', 'makan', 'bergizi', 'gratis', 'program', 'pemerintah',
    'indonesia', 'anak', 'sekolah', 'siswa', 'saja', 'sudah',
    'juga', 'ini', 'itu', 'ada', 'dengan', 'yang', 'dan', 'di',
    'ke', 'dari', 'untuk', 'pada', 'dalam', 'akan', 'tidak', 'bisa'
]
semua_stopword = set(stopword_list) | set(custom_stopwords)

def tokenize_dan_clean(teks):
    if not teks or teks.strip() == '':
        return ''
    tokens = word_tokenize(teks)
    tokens = [t for t in tokens if t not in semua_stopword and len(t) > 2]
    tokens = [stem_kustom(t) for t in tokens]
    return ' '.join(tokens)

# ═══════════════════════════════════════════════════════════════
# 5. PEMROSESAN DATA
# ═══════════════════════════════════════════════════════════════
print('\n⏳ Membersihkan teks...')
t0 = time.time()
df['teks_clean'] = df['teks'].apply(cleaning_teks)
df['teks_proses'] = df['teks_clean'].apply(tokenize_dan_clean)
df = df[df['teks_proses'].str.strip() != '']
df.reset_index(drop=True, inplace=True)
print(f'✅ Cleaning selesai ({len(df)} tweets) dalam {time.time()-t0:.1f}s')

# ═══════════════════════════════════════════════════════════════
# 6. LABELING SENTIMEN (IndoBERT atau Cache)
# ═══════════════════════════════════════════════════════════════
CACHE_FILE = 'data_berlabel.csv'

if SKIP_LABEL_INDOBERT and os.path.exists(CACHE_FILE):
    print('\n⏳ Memuat label dari cache...')
    df_cache = pd.read_csv(CACHE_FILE)
    df['sentimen'] = df_cache['sentimen']
    print(f'✅ Label dimuat dari {CACHE_FILE}')
else:
    print('\n⏳ Loading IndoBERT...')
    t0 = time.time()
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "text-classification",
        model="w11wo/indonesian-roberta-base-sentiment-classifier",
        tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier",
        device=-1  # CPU agar stabil di Streamlit
    )

    from tqdm import tqdm
    teks_list = df['teks_clean'].fillna('').tolist()
    hasil_label = []
    batch_size = 32
    for i in tqdm(range(0, len(teks_list), batch_size)):
        batch = [t[:512] if t.strip() != '' else 'netral' for t in teks_list[i:i+batch_size]]
        try:
            hasil_batch = sentiment_pipeline(batch)
            for h in hasil_batch:
                label = h['label'].lower()
                if label == 'positive':
                    hasil_label.append('Positif')
                elif label == 'negative':
                    hasil_label.append('Negatif')
                else:
                    hasil_label.append('Netral')
        except:
            hasil_label.extend(['Netral'] * len(batch))

    df['sentimen'] = hasil_label
    # Simpan cache
    df_export = df[['teks', 'teks_clean', 'teks_proses', 'tanggal', 'sentimen']]
    df_export.to_csv(CACHE_FILE, index=False)
    print(f'✅ Label selesai ({time.time()-t0:.1f}s) — cache disimpan ke {CACHE_FILE}')

print('\n📊 Distribusi Sentimen:')
print(df['sentimen'].value_counts())
print(df['sentimen'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# ─── Presentase Sentimen ──────────────────────────────────
print('\n📊 PRESENTASE SENTIMEN:')
print('=' * 45)
sentimen_counts = df['sentimen'].value_counts()
total = len(df)
for s in ['Positif', 'Negatif', 'Netral']:
    jml = sentimen_counts.get(s, 0)
    pct = (jml / total) * 100
    bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
    print(f'  {s:<10} {jml:>5} tweet  {pct:>5.1f}%  {bar}')
print('=' * 45)

# ─── WordCloud ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, s in zip(axes, ['Positif', 'Negatif', 'Netral']):
    teks_gabung = ' '.join(df[df['sentimen'] == s]['teks_proses'].dropna())
    if teks_gabung.strip():
        wc = WordCloud(width=400, height=300, background_color='white',
                       colormap={'Positif': 'Greens', 'Negatif': 'Reds', 'Netral': 'Blues'}[s],
                       max_words=80).generate(teks_gabung)
        ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'WordCloud - {s}', fontweight='bold')
plt.tight_layout()
plt.savefig('wordcloud_sentimen.png', dpi=150, bbox_inches='tight')
plt.show()

# ═══════════════════════════════════════════════════════════════
# 7. FITUR SENTIMEN LEXICON + TF-IDF
# ═══════════════════════════════════════════════════════════════
def hitung_fitur_sentimen(teks_series):
    hasil = []
    for teks in teks_series:
        if not isinstance(teks, str) or teks.strip() == '':
            hasil.append([0, 0, 0, 0.0, 0.0, 0.0, 0.5])
            continue
        kata = teks.lower().split()
        pos = sum(1 for k in kata if k in lexicon_positif)
        neg = sum(1 for k in kata if k in lexicon_negatif)
        total_k = pos + neg
        pos_ratio = pos / total_k if total_k > 0 else 0.0
        neg_ratio = neg / total_k if total_k > 0 else 0.0
        polarity = pos - neg
        sentimen_persen = ((pos - neg) / (total_k + 1)) * 100
        sentimen_score = (sentimen_persen / 100 + 1) / 2 if total_k > 0 else 0.5
        hasil.append([pos, neg, polarity, pos_ratio, neg_ratio,
                      round(sentimen_persen, 2), round(sentimen_score, 4)])
    return np.array(hasil)

# ─── Rata-rata Skor Lexicon per Kelas ─────────────────────
print('\n📊 RATA-RATA SKOR LEXICON PER KELAS:')
print('=' * 55)
for s in ['Positif', 'Negatif', 'Netral']:
    tg = ' '.join(df[df['sentimen'] == s]['teks_clean'].dropna())
    if tg.strip():
        kt = tg.lower().split()
        p = sum(1 for k in kt if k in lexicon_positif)
        n = sum(1 for k in kt if k in lexicon_negatif)
        sk = ((p - n) / (p + n + 1)) * 100
        print(f'  {s:<10} → lexicon polarity: {sk:>+6.1f}%')
print('=' * 55)

# ─── TF-IDF + Fitur Sentimen ──────────────────────────────
df_model = df.copy()
X = df_model['teks_clean']
y = df_model['sentimen']

print(f'\n📊 Kelas: {y.value_counts().to_dict()}')
print(f'   Total: {len(X)}')

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                         min_df=2, max_df=0.95, sublinear_tf=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

fitur_sentimen_train = hitung_fitur_sentimen(X_train)
fitur_sentimen_test = hitung_fitur_sentimen(X_test)

X_train_gabung = hstack([X_train_tfidf, csr_matrix(fitur_sentimen_train)])
X_test_gabung = hstack([X_test_tfidf, csr_matrix(fitur_sentimen_test)])

print(f'✅ Dimensi fitur: Train {X_train_gabung.shape}, Test {X_test_gabung.shape}')

# ═══════════════════════════════════════════════════════════════
# 8. TRAINING & EVALUASI MODEL
# ═══════════════════════════════════════════════════════════════
def evaluasi_model(nama, model, X_tr, X_te, y_tr, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_te, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    print(f'  {nama:<25} acc={acc:.4f}  f1={f1:.4f}  ({time.time()-t0:.1f}s)')
    return {'Model': nama, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Score': f1, 'y_pred': y_pred, 'model_obj': model}

hasil_semua = []

print('\n⏳ Melatih model...')
models = [
    ('SVM (Linear)', SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)),
    ('Naive Bayes', MultinomialNB(alpha=1.0)),
]

if not SKIP_MODEL_LAMBAN:
    models.extend([
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('Decision Tree', DecisionTreeClassifier(class_weight='balanced', random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])

for nama, model in models:
    if nama == 'Naive Bayes':
        scaler_nb = MinMaxScaler()
        X_tr_nb = csr_matrix(scaler_nb.fit_transform(X_train_gabung.toarray()))
        X_te_nb = csr_matrix(scaler_nb.transform(X_test_gabung.toarray()))
        h = evaluasi_model(nama, model, X_tr_nb, X_te_nb, y_train, y_test)
    else:
        h = evaluasi_model(nama, model, X_train_gabung, X_test_gabung, y_train, y_test)
    hasil_semua.append({k: v for k, v in h.items() if k not in ['y_pred', 'model_obj']})

# ─── Tabel Komparasi ──────────────────────────────────────
df_komp = pd.DataFrame(hasil_semua).set_index('Model').round(4)
print('\n📊 TABEL KOMPARASI MODEL:')
print(df_komp.to_string())
print(f'\n🏆 Terbaik (Accuracy): {df_komp["Accuracy"].idxmax()}')
print(f'🏆 Terbaik (F1-Score): {df_komp["F1-Score"].idxmax()}')

# ─── MSE ──────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_test)
y_test_num = le.transform(y_test)
print('\n📊 MSE per Model:')
for h in hasil_semua:
    # find the matching full result
    for hr in [x for x in locals().values() if isinstance(x, dict) and x.get('Model') == h['Model']]:
        yp = le.transform(hr['y_pred'])
        mse = ((y_test_num - yp) ** 2).mean()
        print(f'  {h["Model"]:<25}: {mse:.4f}')

# ═══════════════════════════════════════════════════════════════
# 9. CROSS VALIDATION + OVERFIT CHECK (ringkas)
# ═══════════════════════════════════════════════════════════════
print(f'\n⏳ {CV_FOLDS}-fold Cross Validation...')
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

pipelines = {}
for nama, _ in models:
    if nama == 'Naive Bayes':
        p = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                      ('clf', MultinomialNB())])
    else:
        clf = {'SVM (Linear)': SVC(kernel='linear', class_weight='balanced', random_state=42),
               'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
               'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
               'KNN': KNeighborsClassifier(n_neighbors=5),
               'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
               'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)}[nama]
        p = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
                      ('clf', clf)])
    pipelines[nama] = p

# Gunakan X_stem untuk CV (pipeline internal)
X_stem = df_model['teks_proses']
for nama, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X_stem, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f'  {nama:<25}: {scores.mean():.4f} ± {scores.std():.4f}')

# ═══════════════════════════════════════════════════════════════
# 10. HYPERPARAMETER TUNING (Opsional)
# ═══════════════════════════════════════════════════════════════
if not SKIP_OPTUNA:
    print('\n⏳ Menjalankan Optuna...')
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        mf = trial.suggest_categorical('tfidf__max_features', [3000, 5000])
        nm = trial.suggest_int('tfidf__ngram_max', 1, 2)
        md = trial.suggest_int('tfidf__min_df', 1, 2)
        C = trial.suggest_float('clf__C', 0.1, 10, log=True)
        p = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=mf, ngram_range=(1, nm), min_df=md, sublinear_tf=True)),
            ('clf', SVC(kernel='linear', C=C, class_weight='balanced', random_state=42))
        ])
        return cross_val_score(p, X_stem, y, cv=3, scoring='f1_weighted', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    print(f'✅ Optuna selesai — Best F1: {study.best_value:.4f}')
    print(f'   Parameters: {study.best_params}')

    # Evaluasi model tuned
    bp = study.best_params
    best_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=bp['tfidf__max_features'],
                                   ngram_range=(1, bp['tfidf__ngram_max']),
                                   min_df=bp['tfidf__min_df'], sublinear_tf=True)),
        ('clf', SVC(kernel='linear', C=bp['clf__C'], class_weight='balanced', random_state=42))
    ])
    best_model.fit(X_train, y_train)
    yp = best_model.predict(X_test)
    print(f'  Test Accuracy: {accuracy_score(y_test, yp):.4f}')
    print(f'  Test F1: {f1_score(y_test, yp, average="weighted"):.4f}')

    with open('svm_tuned_mbg.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print('✅ Model tersimpan: svm_tuned_mbg.pkl')
else:
    print('\n⏭️ Optuna dilewati (SKIP_OPTUNA = True)')

# ═══════════════════════════════════════════════════════════════
# 11. CONTOH PREDIKSI
# ═══════════════════════════════════════════════════════════════
print('\n🔍 Contoh Prediksi:')
teks_baru = [
    "program makan bergizi gratis sangat membantu anak sekolah",
    "banyak anak keracunan makanan di sekolah",
    "sudah dapat info program mbg besok mulai dibagikan"
]

# Gunakan model SVM (terbaik dari hasil training)
svm_best = None
for hh in hasil_semua:
    for hr in [x for x in locals().values() if isinstance(x, dict) and x.get('Model') == hh['Model']]:
        if hh['Model'] == 'SVM (Linear)':
            svm_best = hr['model_obj']
            break
    if svm_best:
        break

if svm_best is None:
    svm_best = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
    svm_best.fit(X_train_gabung, y_train)

for teks in teks_baru:
    # Prediksi langsung pakai pipeline lengkap (TF-IDF + sentimen fitur)
    teks_clean = cleaning_teks(teks)
    teks_vec = tfidf.transform([teks_clean])
    teks_sent = hitung_fitur_sentimen([teks_clean])
    teks_gab = hstack([teks_vec, csr_matrix(teks_sent)])
    pred = svm_best.predict(teks_gab)[0]

    kata = teks.lower().split()
    p = sum(1 for k in kata if k in lexicon_positif)
    n = sum(1 for k in kata if k in lexicon_negatif)
    skor = ((p - n) / (p + n + 1)) * 100
    arah = '🟢 positif' if skor > 0 else ('🔴 negatif' if skor < 0 else '⚪ netral')
    print(f'  Teks    : "{teks}"')
    print(f'  Prediksi: {pred}  |  Lexicon: {skor:+.1f}% ({arah})')
    print()

print('\n✅ Selesai!')
