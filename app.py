import streamlit as st
import pandas as pd
import time
import torch
import os
from collections import Counter
from transformers import pipeline
from google import genai

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Multi-LLM Sentiment Analysis Bot",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    background-color: #f0f2f6;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# LOAD HUGGING FACE MODEL 1
# ---------------------------------------------------

@st.cache_resource
def load_model_1():

    model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1
    )

    return model


# ---------------------------------------------------
# LOAD HUGGING FACE MODEL 2
# ---------------------------------------------------

@st.cache_resource
def load_model_2():

    model = pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
        device=0 if torch.cuda.is_available() else -1
    )

    return model


# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

with st.spinner("Loading AI models..."):

    model_1 = load_model_1()
    model_2 = load_model_2()


# ---------------------------------------------------
# CONVERT 5 SENTIMENTS INTO 3 SENTIMENTS
# ---------------------------------------------------

def convert_to_three_sentiments(label):

    label = label.lower()

    if "positive" in label:
        return "Positive"

    elif "negative" in label:
        return "Negative"

    else:
        return "Neutral"


# ---------------------------------------------------
# MODEL 1 PREDICTION
# ---------------------------------------------------

def predict_roberta(text):

    try:

        start_time = time.time()

        result = model_1(text)[0]

        sentiment = result["label"].capitalize()

        confidence = round(
            result["score"] * 100,
            2
        )

        processing_time = round(
            time.time() - start_time,
            4
        )

        return {
            "Model": "RoBERTa",
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Time": processing_time
        }

    except Exception as e:

        return {
            "Model": "RoBERTa",
            "Sentiment": "Error",
            "Confidence": 0,
            "Time": 0
        }


# ---------------------------------------------------
# MODEL 2 PREDICTION
# ---------------------------------------------------

def predict_multilingual(text):

    try:

        start_time = time.time()

        result = model_2(text)[0]

        sentiment = convert_to_three_sentiments(
            result["label"]
        )

        confidence = round(
            result["score"] * 100,
            2
        )

        processing_time = round(
            time.time() - start_time,
            4
        )

        return {
            "Model": "Multilingual Model",
            "Sentiment": sentiment,
            "Confidence": confidence,
            "Time": processing_time
        }

    except Exception as e:

        return {
            "Model": "Multilingual Model",
            "Sentiment": "Error",
            "Confidence": 0,
            "Time": 0
        }


# ---------------------------------------------------
# GEMINI CLIENT
# ---------------------------------------------------

def get_gemini_client():

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        client = genai.Client(
            api_key=api_key
        )

        return client

    except:

        return None


# ---------------------------------------------------
# GEMINI PREDICTION
# ---------------------------------------------------

def predict_gemini(text):

    client = get_gemini_client()

    if client is None:

        return {
            "Model": "Gemini",
            "Sentiment": "API Key Missing",
            "Confidence": "N/A",
            "Time": 0
        }

    try:

        start_time = time.time()

        prompt = f"""
        Analyze the sentiment of the following text.

        Text:
        {text}

        Classify the sentiment as exactly one of:

        Positive
        Negative
        Neutral

        Return only one word.
        Do not give any explanation.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        sentiment = response.text.strip().capitalize()

        if sentiment not in [
            "Positive",
            "Negative",
            "Neutral"
        ]:

            sentiment = "Neutral"

        processing_time = round(
            time.time() - start_time,
            4
        )

        return {
            "Model": "Gemini",
            "Sentiment": sentiment,
            "Confidence": "N/A",
            "Time": processing_time
        }

    except Exception as e:

        return {
            "Model": "Gemini",
            "Sentiment": "Error",
            "Confidence": "N/A",
            "Time": 0
        }


# ---------------------------------------------------
# MAJORITY VOTING
# ---------------------------------------------------

def majority_vote(results):

    valid_sentiments = [
        result["Sentiment"]
        for result in results
        if result["Sentiment"]
        in ["Positive", "Negative", "Neutral"]
    ]

    if not valid_sentiments:

        return "Unable to Predict"

    vote_count = Counter(valid_sentiments)

    return vote_count.most_common(1)[0][0]


# ---------------------------------------------------
# SENTIMENT EMOJI
# ---------------------------------------------------

def get_sentiment_emoji(sentiment):

    if sentiment == "Positive":
        return "😊"

    elif sentiment == "Negative":
        return "😞"

    elif sentiment == "Neutral":
        return "😐"

    else:
        return "❓"


# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 Multi-LLM Sentiment Analysis Bot</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Compare sentiment predictions from RoBERTa,
    a Multilingual AI Model, and Gemini
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("About the Project")

    st.write("""
    This GenAI bot analyzes text using multiple AI models.
    """)

    st.write("Models used:")

    st.write("1. RoBERTa")
    st.write("2. Multilingual Sentiment Model")
    st.write("3. Gemini LLM")

    st.divider()

    st.write("""
    The final result is generated using majority voting.
    """)


# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

user_text = st.text_area(
    "Enter text for sentiment analysis:",
    placeholder="Example: I really enjoyed this product, but the delivery was slow.",
    height=150
)


# ---------------------------------------------------
# EXAMPLE TEXT
# ---------------------------------------------------

example = st.selectbox(
    "Or select an example:",
    [
        "Select an example",
        "I absolutely loved this product!",
        "The service was terrible and disappointing.",
        "The product is okay, nothing special.",
        "The movie was good but the ending was disappointing."
    ]
)

if example != "Select an example":

    user_text = example


# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True,
    type="primary"
):

    if not user_text.strip():

        st.warning(
            "Please enter some text before analyzing."
        )

    else:

        with st.spinner(
            "Multiple AI models are analyzing your text..."
        ):

            result_1 = predict_roberta(user_text)

            result_2 = predict_multilingual(user_text)

            result_3 = predict_gemini(user_text)

            all_results = [
                result_1,
                result_2,
                result_3
            ]

            final_sentiment = majority_vote(
                all_results
            )


        # -------------------------------------------
        # FINAL RESULT
        # -------------------------------------------

        st.divider()

        st.subheader("🎯 Final Sentiment")

        emoji = get_sentiment_emoji(
            final_sentiment
        )

        st.markdown(
            f"""
            <div class="result-box">
            {emoji} {final_sentiment}
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------
        # MODEL RESULTS
        # -------------------------------------------

        st.subheader("🧠 Individual Model Results")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "RoBERTa",
                f"{get_sentiment_emoji(result_1['Sentiment'])} "
                f"{result_1['Sentiment']}"
            )

            st.write(
                f"Confidence: {result_1['Confidence']}%"
            )

            st.write(
                f"Time: {result_1['Time']} seconds"
            )


        with col2:

            st.metric(
                "Multilingual Model",
                f"{get_sentiment_emoji(result_2['Sentiment'])} "
                f"{result_2['Sentiment']}"
            )

            st.write(
                f"Confidence: {result_2['Confidence']}%"
            )

            st.write(
                f"Time: {result_2['Time']} seconds"
            )


        with col3:

            st.metric(
                "Gemini",
                f"{get_sentiment_emoji(result_3['Sentiment'])} "
                f"{result_3['Sentiment']}"
            )

            st.write(
                "Confidence: Not provided by Gemini"
            )

            st.write(
                f"Time: {result_3['Time']} seconds"
            )


        # -------------------------------------------
        # RESULTS TABLE
        # -------------------------------------------

        st.subheader("📋 Complete Model Comparison")

        results_df = pd.DataFrame(
            all_results
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------
        # PROCESSING TIME CHART
        # -------------------------------------------

        st.subheader(
            "⏱️ Processing Time Comparison"
        )

        time_df = results_df[
            ["Model", "Time"]
        ].set_index("Model")

        st.bar_chart(time_df)


        # -------------------------------------------
        # VOTING RESULTS
        # -------------------------------------------

        st.subheader(
            "🗳️ Majority Voting"
        )

        sentiment_counts = (
            results_df[
                results_df["Sentiment"].isin(
                    [
                        "Positive",
                        "Negative",
                        "Neutral"
                    ]
                )
            ]["Sentiment"]
            .value_counts()
        )

        st.bar_chart(sentiment_counts)


        # -------------------------------------------
        # EXPLANATION
        # -------------------------------------------

        st.info(
            f"""
            The final sentiment is **{final_sentiment}** because
            the system compares predictions from all available
            models and selects the sentiment with the most votes.
            """
        )
