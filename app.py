import streamlit as st
import pandas as pd
import torch
import time

from transformers import pipeline


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Happiness GenAI Bot",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_CONFIG = {

    "RoBERTa 3-Class": {
        "task": "sentiment-analysis",
        "model_id":
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    },

    "Multilingual Sentiment": {
        "task": "text-classification",
        "model_id":
        "tabularisai/multilingual-sentiment-analysis"
    },

    "BERTweet Sentiment": {
        "task": "sentiment-analysis",
        "model_id":
        "finiteautomata/bertweet-base-sentiment-analysis"
    },

    "DistilBERT SST-2": {
        "task": "sentiment-analysis",
        "model_id":
        "distilbert-base-uncased-finetuned-sst-2-english"
    },

    "DistilRoBERTa Emotion": {
        "task": "text-classification",
        "model_id":
        "j-hartmann/emotion-english-distilroberta-base"
    }
}


# ==========================================
# LOAD SELECTED MODEL
# ==========================================

@st.cache_resource
def load_model(model_name):

    config = MODEL_CONFIG[
        model_name
    ]

    device = (
        0
        if torch.cuda.is_available()
        else -1
    )

    return pipeline(

        config["task"],

        model=config[
            "model_id"
        ],

        device=device,

        truncation=True
    )


# ==========================================
# SENTIMENT NORMALIZATION
# ==========================================

def normalize_sentiment(label):

    label = str(
        label
    ).lower()

    if (
        label in [
            "label_2",
            "pos",
            "positive"
        ]
        or "positive" in label
    ):
        return "Positive"

    if (
        label in [
            "label_0",
            "neg",
            "negative"
        ]
        or "negative" in label
    ):
        return "Negative"

    return "Neutral"


# ==========================================
# EMOTION DETECTION
# ==========================================

def detect_emotion(
    label,
    text,
    sentiment=None
):

    label = str(
        label
    ).lower()

    text = str(
        text
    ).lower()

    angry_words = [
        "angry",
        "furious",
        "unacceptable",
        "worst",
        "frustrated",
        "terrible",
        "awful"
    ]

    surprise_words = [
        "surprise",
        "surprised",
        "unexpected",
        "wow",
        "did not expect"
    ]

    if label in [
        "anger",
        "disgust"
    ]:
        return "Angry"

    if any(
        word in text
        for word in angry_words
    ):
        return "Angry"

    if label == "surprise":
        return "Surprise"

    if any(
        word in text
        for word in surprise_words
    ):
        return "Surprise"

    if label in [
        "joy",
        "love"
    ]:
        return "Happy"

    if label in [
        "sadness",
        "fear"
    ]:
        return "Sad"

    if sentiment == "Positive":
        return "Happy"

    if sentiment == "Negative":
        return "Sad"

    return "Surprise"


# ==========================================
# ANALYZE SINGLE REVIEW
# ==========================================

def analyze_review(
    text,
    model_name
):

    model = load_model(
        model_name
    )

    start_time = time.time()

    result = model(
        text[:1500]
    )[0]

    raw_label = result[
        "label"
    ]

    confidence = round(
        float(
            result["score"]
        ) * 100,
        2
    )

    if (
        model_name
        == "DistilRoBERTa Emotion"
    ):

        emotion = detect_emotion(
            raw_label,
            text
        )

        if emotion == "Happy":
            sentiment = "Positive"

        elif emotion in [
            "Sad",
            "Angry"
        ]:
            sentiment = "Negative"

        else:
            sentiment = "Neutral"

    else:

        sentiment = (
            normalize_sentiment(
                raw_label
            )
        )

        emotion = detect_emotion(
            raw_label,
            text,
            sentiment
        )

    customer_happy = (
        "Yes"
        if emotion in [
            "Happy",
            "Surprise"
        ]
        else "No"
    )

    processing_time = round(
        time.time()
        - start_time,
        4
    )

    return {

        "Sentiment":
        sentiment,

        "Emotion":
        emotion,

        "Customer Happy":
        customer_happy,

        "Confidence":
        confidence,

        "Processing Time":
        processing_time
    }


# ==========================================
# TITLE
# ==========================================

st.title(
    "🤖 Customer Happiness & Review GenAI Bot"
)

st.write(
    """
    Analyze customer feedback and reviews
    using five Hugging Face AI models.
    """
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    "Model Selection"
)

selected_model = (
    st.sidebar.selectbox(

        "Select AI Model",

        list(
            MODEL_CONFIG.keys()
        )
    )
)

st.sidebar.success(
    f"Selected: {selected_model}"
)

st.sidebar.info(
    """
    Only the selected model is loaded.
    This reduces memory usage during deployment.
    """
)


# ==========================================
# TABS
# ==========================================

tab1, tab2 = st.tabs([

    "🤖 Single Review Bot",

    "📊 Upload Dataset & Report"
])


# ==========================================
# TAB 1
# ==========================================

with tab1:

    st.header(
        "Analyze One Customer Review"
    )

    customer_text = (
        st.text_area(

            "Enter Customer Feedback or Review",

            height=180,

            placeholder=
            "Example: I am very angry because the product arrived damaged."
        )
    )

    if st.button(
        "Analyze Customer",
        type="primary"
    ):

        if not customer_text.strip():

            st.warning(
                "Please enter a customer review."
            )

        else:

            with st.spinner(
                f"Analyzing with {selected_model}..."
            ):

                result = analyze_review(

                    customer_text,

                    selected_model
                )

            col1, col2 = st.columns(2)

            col3, col4 = st.columns(2)

            col1.metric(
                "Sentiment",
                result["Sentiment"]
            )

            col2.metric(
                "Customer Emotion",
                result["Emotion"]
            )

            col3.metric(
                "Confidence",
                f"{result['Confidence']}%"
            )

            col4.metric(
                "Processing Time",
                f"{result['Processing Time']} sec"
            )

            if (
                result[
                    "Customer Happy"
                ]
                == "Yes"
            ):

                st.success(
                    "😊 Customer is likely happy."
                )

            else:

                st.error(
                    "⚠️ Customer is not happy and may need attention."
                )


# ==========================================
# TAB 2
# ==========================================

with tab2:

    st.header(
        "Upload Customer Dataset"
    )

    uploaded_file = (
        st.file_uploader(

            "Upload CSV File",

            type=["csv"]
        )
    )

    if uploaded_file:

        df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"""
            Dataset uploaded successfully:
            {len(df)} customers
            """
        )

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        columns = list(
            df.columns
        )

        feedback_index = (
            columns.index("feedback")
            if "feedback" in columns
            else 0
        )

        review_index = (
            columns.index("review")
            if "review" in columns
            else 0
        )

        feedback_column = (
            st.selectbox(

                "Select Feedback Column",

                columns,

                index=feedback_index
            )
        )

        review_column = (
            st.selectbox(

                "Select Review Column",

                columns,

                index=review_index
            )
        )

        rows_to_analyze = (
            st.number_input(

                "Number of Customers to Analyze",

                min_value=1,

                max_value=len(df),

                value=min(
                    500,
                    len(df)
                ),

                step=100
            )
        )


        if st.button(
            "Generate Complete Customer Report",
            type="primary"
        ):

            work_df = df.head(
                int(rows_to_analyze)
            ).copy()

            texts = (

                work_df[
                    feedback_column
                ]

                .fillna("")

                .astype(str)

                + " "

                + work_df[
                    review_column
                ]

                .fillna("")

                .astype(str)

            ).tolist()


            with st.spinner(
                "Analyzing customer reviews..."
            ):

                model = load_model(
                    selected_model
                )

                outputs = model(

                    [
                        text[:1500]
                        for text in texts
                    ],

                    batch_size=16,

                    truncation=True,

                    max_length=256
                )


            sentiments = []

            emotions = []

            confidences = []

            happy_predictions = []


            for text, output in zip(
                texts,
                outputs
            ):

                raw_label = output[
                    "label"
                ]

                confidence = round(

                    float(
                        output["score"]
                    ) * 100,

                    2
                )


                if (
                    selected_model
                    == "DistilRoBERTa Emotion"
                ):

                    emotion = (
                        detect_emotion(

                            raw_label,

                            text
                        )
                    )

                    if emotion == "Happy":

                        sentiment = "Positive"

                    elif emotion in [
                        "Sad",
                        "Angry"
                    ]:

                        sentiment = "Negative"

                    else:

                        sentiment = "Neutral"


                else:

                    sentiment = (
                        normalize_sentiment(
                            raw_label
                        )
                    )

                    emotion = (
                        detect_emotion(

                            raw_label,

                            text,

                            sentiment
                        )
                    )


                happy = (
                    "Yes"
                    if emotion in [
                        "Happy",
                        "Surprise"
                    ]
                    else "No"
                )


                sentiments.append(
                    sentiment
                )

                emotions.append(
                    emotion
                )

                confidences.append(
                    confidence
                )

                happy_predictions.append(
                    happy
                )


            work_df[
                "predicted_sentiment"
            ] = sentiments

            work_df[
                "predicted_emotion"
            ] = emotions

            work_df[
                "confidence"
            ] = confidences

            work_df[
                "predicted_customer_happy"
            ] = happy_predictions


            emotion_counts = (

                work_df[
                    "predicted_emotion"
                ]

                .value_counts()

                .reindex(

                    [
                        "Happy",
                        "Sad",
                        "Angry",
                        "Surprise"
                    ],

                    fill_value=0
                )
            )


            total = len(
                work_df
            )


            report = pd.DataFrame({

                "Emotion":
                emotion_counts.index,

                "Customers":
                emotion_counts.values,

                "Percentage":
                (
                    emotion_counts.values
                    / total
                    * 100
                ).round(2)
            })


            st.header(
                "📊 Complete Customer Report"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            col1.metric(
                "😊 Happy",
                int(
                    emotion_counts[
                        "Happy"
                    ]
                )
            )

            col2.metric(
                "😢 Sad",
                int(
                    emotion_counts[
                        "Sad"
                    ]
                )
            )

            col3.metric(
                "😠 Angry",
                int(
                    emotion_counts[
                        "Angry"
                    ]
                )
            )

            col4.metric(
                "😲 Surprise",
                int(
                    emotion_counts[
                        "Surprise"
                    ]
                )
            )


            st.subheader(
                "Emotion Summary"
            )

            st.dataframe(

                report,

                use_container_width=True,

                hide_index=True
            )


            st.subheader(
                "Customer Emotion Visualization"
            )

            st.bar_chart(

                report.set_index(
                    "Emotion"
                )["Customers"]
            )


            st.subheader(
                "Happy vs Not Happy"
            )

            happy_counts = (

                work_df[
                    "predicted_customer_happy"
                ]

                .value_counts()
            )

            st.bar_chart(
                happy_counts
            )


            if "product" in work_df.columns:

                st.subheader(
                    "Emotion by Product"
                )

                product_report = (
                    pd.crosstab(

                        work_df[
                            "product"
                        ],

                        work_df[
                            "predicted_emotion"
                        ]
                    )
                )

                st.dataframe(

                    product_report,

                    use_container_width=True
                )


            if "category" in work_df.columns:

                st.subheader(
                    "Emotion by Category"
                )

                category_report = (
                    pd.crosstab(

                        work_df[
                            "category"
                        ],

                        work_df[
                            "predicted_emotion"
                        ]
                    )
                )

                st.dataframe(

                    category_report,

                    use_container_width=True
                )


            if "region" in work_df.columns:

                st.subheader(
                    "Emotion by Region"
                )

                region_report = (
                    pd.crosstab(

                        work_df[
                            "region"
                        ],

                        work_df[
                            "predicted_emotion"
                        ]
                    )
                )

                st.dataframe(

                    region_report,

                    use_container_width=True
                )


            st.subheader(
                "Complete Analyzed Customer Records"
            )

            st.dataframe(

                work_df,

                use_container_width=True
            )


            st.download_button(

                "Download Complete Customer Report",

                work_df.to_csv(
                    index=False
                ).encode("utf-8"),

                "complete_customer_report.csv",

                "text/csv"
            )


            st.download_button(

                "Download Emotion Summary",

                report.to_csv(
                    index=False
                ).encode("utf-8"),

                "emotion_summary_report.csv",

                "text/csv"
            )
