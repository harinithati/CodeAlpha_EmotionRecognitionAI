import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Emotion Recognition AI",
    page_icon="🎤",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = tf.keras.models.load_model(
    "emotion_model.h5"
)

encoder = pickle.load(
    open("label_encoder.pkl", "rb")
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background: linear-gradient(
        to bottom right,
        #0E1117,
        #111827
    );
}

.title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    background: linear-gradient(
        90deg,
        #00C6FF,
        #0072FF
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    color: #B0B0B0;
    font-size: 20px;
    margin-bottom: 40px;
}

.card {
    background-color: #161B22;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(0,198,255,0.2);
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(
        135deg,
        #1E293B,
        #0F172A
    );
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,198,255,0.15);
}

.footer {
    text-align:center;
    color:gray;
    margin-top:40px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------

st.markdown(
    '<div class="title">🎤 Emotion Recognition AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced Deep Learning Speech Emotion Detection System</div>',
    unsafe_allow_html=True
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🧠 AI Dashboard")

st.sidebar.info("""
### 📌 Project Overview

This AI system analyzes speech audio and predicts human emotions using Deep Learning.

### 🚀 Technologies
- TensorFlow
- CNN + LSTM
- Librosa
- Streamlit
- Plotly

### 🎯 Features
✅ Audio Emotion Detection  
✅ AI-Based Prediction  
✅ Waveform Visualization  
✅ Emotion Analytics  
✅ Interactive Dashboard
""")

# -----------------------------------
# FEATURE EXTRACTION
# -----------------------------------

def extract_features(file):

    audio, sample_rate = librosa.load(
        file,
        duration=3,
        offset=0.5
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sample_rate
    )

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate
    )

    mfcc_scaled = np.mean(mfcc.T, axis=0)

    chroma_scaled = np.mean(chroma.T, axis=0)

    mel_scaled = np.mean(mel.T, axis=0)

    features = np.hstack([
        mfcc_scaled,
        chroma_scaled,
        mel_scaled
    ])

    return features, audio, sample_rate

# -----------------------------------
# MAIN LAYOUT
# -----------------------------------

col1, col2 = st.columns([2,1])

with col1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "🎵 Upload WAV Audio File",
        type=["wav"]
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="metric-card">
        <h2>🤖 AI Model</h2>
        <h3>CNN + LSTM</h3>
        <p>Speech Emotion Recognition</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------
# PREDICTION
# -----------------------------------

if uploaded_file is not None:

    st.audio(uploaded_file)

    with st.spinner("🔍 AI Analyzing Emotion..."):

        features, audio, sample_rate = extract_features(
            uploaded_file
        )

        features = features.reshape(
            1,
            features.shape[0],
            1
        )

        prediction = model.predict(features)

        predicted_label = np.argmax(prediction)

        emotion = encoder.inverse_transform(
            [predicted_label]
        )[0]

        confidence = np.max(prediction)

    # -----------------------------------
    # RESULT SECTION
    # -----------------------------------

    st.markdown("---")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(
            f"""
            <div class="metric-card">
            <h2>🎯 Predicted Emotion</h2>
            <h1>{emotion.upper()}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="metric-card">
            <h2>📊 Confidence Score</h2>
            <h1>{confidence:.2%}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------
    # AUDIO WAVEFORM
    # -----------------------------------

    with result_col2:

        fig, ax = plt.subplots(
            figsize=(8,3)
        )

        librosa.display.waveshow(
            audio,
            sr=sample_rate,
            ax=ax
        )

        ax.set_title("Audio Waveform")

        st.pyplot(fig)

    # -----------------------------------
    # PROBABILITY ANALYSIS
    # -----------------------------------

    st.markdown("## 📈 Emotion Probability Analysis")

    emotions = encoder.classes_

    probabilities = prediction[0]

    df = pd.DataFrame({
        "Emotion": emotions,
        "Probability": probabilities
    })

    fig2 = px.bar(
        df,
        x="Emotion",
        y="Probability",
        color="Probability",
        text_auto='.2f',
        title="Emotion Confidence Distribution"
    )

    fig2.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # -----------------------------------
    # INSIGHTS SECTION
    # -----------------------------------

    st.markdown("## 🧠 AI Insights")

    st.info(f"""
    The uploaded speech audio was analyzed using Deep Learning feature extraction techniques including:
    
    ✅ MFCC Features  
    ✅ Chroma Features  
    ✅ Mel Spectrogram Features  
    
    The CNN + LSTM architecture identified the dominant emotion as **{emotion.upper()}** with a confidence score of **{confidence:.2%}**.
    """)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown("""
<div class="footer">

Developed with ❤️ using Deep Learning & Streamlit<br> AI Internship Project

</div>
""", unsafe_allow_html=True)
