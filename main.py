import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gensim.models import Word2Vec
import os

# -----------------------------
# Function: Average Word2Vec
# -----------------------------
def get_average_vector(sentence, model):
    # Handle both list (tokenized) and string input
    if isinstance(sentence, list):
        tokens = sentence
    else:
        tokens = sentence.lower().split()

    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# -----------------------------
# Training Phase (only if models not saved)
# -----------------------------
if not (os.path.exists("spam_word2vec.model") and os.path.exists("spam_classifier_model.pkl") and os.path.exists("label_encoder.pkl")):
    st.write("🔄 Training model... Please wait.")

    # 1️⃣ Load dataset
    df = pd.read_excel("Mail_New1.xlsx")
    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

    # 2️⃣ Encode target labels
    label_encoder = LabelEncoder()
    df['v1'] = label_encoder.fit_transform(df['v1'])  # ham=0, spam=1

    # 3️⃣ Define features and labels
    X = df['v2'].astype(str)
    Y = df['v1']

    # 4️⃣ Tokenize sentences
    X_tokenized = X.apply(lambda x: x.lower().split())

    # 5️⃣ Train Word2Vec
    w2v_model = Word2Vec(sentences=X_tokenized, vector_size=100, window=5, min_count=1, workers=4, sg=1)

    # 6️⃣ Convert text to vectors
    X_vectors = np.array([get_average_vector(text, w2v_model) for text in X_tokenized])

    # 7️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, Y, test_size=0.2, random_state=42)

    # 8️⃣ Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 9️⃣ Evaluate
    pred = model.predict(X_test)
    st.write("✅ Training Complete")
    st.write("Accuracy:", accuracy_score(y_test, pred))
    st.text("Classification Report:\n" + classification_report(y_test, pred))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, pred)))

    # 🔟 Save models
    w2v_model.save("spam_word2vec.model")
    joblib.dump(model, "spam_classifier_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

# -----------------------------
# Load Models for Prediction
# -----------------------------
w2v_model = Word2Vec.load("spam_word2vec.model")
model = joblib.load("spam_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spam Mail Classifier", page_icon="📧", layout="centered")
st.title("📧 Spam Mail Classifier")
st.write("This app uses **Word2Vec + Logistic Regression** to classify emails as **Ham (Not Spam)** or **Spam**.")

# Input text box
user_input = st.text_area("✍️ Enter your email/message:", placeholder="Type or paste an email here...")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        input_vector = get_average_vector(user_input, w2v_model).reshape(1, -1)
        prediction = model.predict(input_vector)[0]
        pred_label = label_encoder.inverse_transform([prediction])[0]

        if pred_label == "spam":
            st.error("🚨 This message is classified as **SPAM**")
        else:
            st.success("✅ This message is classified as **HAM (Not Spam)**")
