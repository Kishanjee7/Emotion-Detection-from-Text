import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title
st.title("🧠 Emotion Detection from Text")
st.markdown("Enter a sentence below, and the model will predict the emotion.")

# Input box
user_input = st.text_area("Type a sentence:", "")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        
        emoji_map = {
            "joy": "😊", "sadness": "😢", "anger": "😠",
            "love": "❤️", "fear": "😨", "surprise": "😲"
        }
        
        emoji = emoji_map.get(prediction, "❓")
        st.success(f"**Detected Emotion: {prediction.capitalize()} {emoji}**")
