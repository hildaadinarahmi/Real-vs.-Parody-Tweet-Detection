import streamlit as st
import joblib
import numpy as np
from textblob import TextBlob
from scipy.sparse import hstack
import shap

# Load model dan vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
engineered_feature_names = joblib.load("engineered_features.pkl")

# Fitur engineering fungsi
def extract_features(tweet):
    word_count = len(tweet.split())
    avg_word_len = np.mean([len(w) for w in tweet.split()]) if word_count > 0 else 0
    exclamation_count = tweet.count("!")
    question_count = tweet.count("?")
    sentiment = TextBlob(tweet).sentiment.polarity
    return [word_count, avg_word_len, exclamation_count, question_count, sentiment]

# UI
st.title("🔍 Real vs. Parody Tweet Classifier")
st.write("Masukkan tweet dari figur publik, dan kami prediksi apakah ini asli atau parodi.")

user_input = st.text_area("Masukkan tweet di sini:")

if user_input:
    tfidf_features = vectorizer.transform([user_input])
    engineered = np.array(extract_features(user_input)).reshape(1, -1)
    final_input = hstack([tfidf_features, engineered])

    pred = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0][1]

    st.markdown(f"### 🧠 Prediksi: {'🟢 Real' if pred else '🔴 Parody'}")
    st.markdown(f"**Confidence:** {proba:.2f}")

    explainer = shap.LinearExplainer(model, final_input, feature_perturbation="interventional")
    shap_values = explainer.shap_values(final_input)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("#### 🔎 Interpretasi SHAP:")
    shap.summary_plot(shap_values, final_input,
                      feature_names=vectorizer.get_feature_names_out().tolist() + engineered_feature_names)
    st.pyplot(bbox_inches='tight')