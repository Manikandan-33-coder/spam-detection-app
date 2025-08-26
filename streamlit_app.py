import streamlit as st
import joblib
import numpy as np
from gensim.models import Word2Vec

# -----------------------------
# Load Models
# -----------------------------
w2v_model = Word2Vec.load("spam_word2vec.model")
model = joblib.load("spam_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Helper function to vectorize input
# -----------------------------
def get_average_vector(sentence, model):
    tokens = sentence.lower().split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧", layout="centered")

st.title("📧 Spam Mail Classifier")
st.write("This app uses **Word2Vec + Logistic Regression** to classify emails as **Ham (Not Spam)** or **Spam**.")

# Text input
user_input = st.text_area("✍️ Enter your email/message:", placeholder="Type or paste an email here...")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        # Convert input to vector
        input_vector = get_average_vector(user_input, w2v_model).reshape(1, -1)

        # Predict
        prediction = model.predict(input_vector)[0]
        pred_label = label_encoder.inverse_transform([prediction])[0]

        # Display result
        if pred_label == "spam":
            st.error("🚨 This message is classified as **SPAM**")
        else:
            st.success("✅ This message is classified as **HAM (Not Spam)**")
