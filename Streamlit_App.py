import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import joblib
import seaborn as sns
import requests
import pickle
import os
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    classification_report, precision_recall_curve, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BreastCancerDetectionApp:
    def __init__(self):
        st.set_page_config(
            page_title="AI Breast Cancer Diagnostic Assistant", 
            page_icon="🩺", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
        self.setup_page()
        self.load_resources()

    def setup_page(self):
        # Option to change background color
        bg_color = st.sidebar.selectbox("Select Background Color", ["Grey", "White", "Light Blue", "Light Green", "Light Yellow", "Light Grey", "Light Pink"])
        if bg_color == "Grey":
            bg_color_code = "#F4F6F7"
        elif bg_color == "White":
            bg_color_code = "#FFFFFF"
        elif bg_color == "Light Blue":
            bg_color_code = "#E3F2FD"
        elif bg_color == "Light Green":
            bg_color_code = "#E8F5E9"
        elif bg_color == "Light Yellow":
            bg_color_code = "#FFFDE7"
        elif bg_color == "Light Grey":
            bg_color_code = "#F5F5F5"
        elif bg_color == "Light Pink":
            bg_color_code = "#FCE4EC"

        st.markdown(f"""
        <style>
        .main-header {{
            background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .stApp {{
            background-color: {bg_color_code};
        }}
        .metric-container {{
            background: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            border: 1px solid #E0E0E0;
        }}
        .metric-container:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .prediction-card {{
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .developer-section {{
            background-color: #F8F9F9;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 24px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding: 20px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }}
        </style>
        """, unsafe_allow_html=True)

    def load_resources(self):

        @st.cache_resource
        def load_model_data():
            import os
            import pandas as pd
            import pickle
            import numpy as np

            # Paths
            model_path = "Weight files/adaboost_model_with_smote_on_original_data.pkl"
            scaler_path = "Weight files/scaler.pkl"
            dataset_path = "breast_cancer_data.csv"

            # load dataset first (we will split locally)
            if not os.path.exists(dataset_path):
                st.error(f"❌ Dataset file not found at: {dataset_path}")
                return None
            data = pd.read_csv(dataset_path)
            # drop stray unnamed columns
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

            # Load model
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found at: {model_path}")
                return None

            with open(model_path, "rb") as f:
                obj = pickle.load(f)

            # Attempt to extract estimator if wrapped in container
            model = None
            if hasattr(obj, "predict") and hasattr(obj, "predict_proba"):
                model = obj
            else:
                # Common containers
                if isinstance(obj, dict):
                    for key in ["model", "clf", "estimator", "classifier"]:
                        if key in obj and hasattr(obj[key], "predict"):
                            model = obj[key]
                            break
                if model is None and (isinstance(obj, (list, tuple)) or isinstance(obj, np.ndarray)):
                    for el in obj:
                        if hasattr(el, "predict") and hasattr(el, "predict_proba"):
                            model = el
                            break

            if model is None:
                st.error("❌ Loaded pickle does not contain a usable estimator. Please recreate the model file.")
                return None

            # Load scaler if present, otherwise will fit a new scaler later
            scaler = None
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            else:
                st.warning(f"⚠️ Scaler not found at {scaler_path}. A new scaler will be fit from training data.")

            # Prepare feature names (exclude id/diagnosis)
            feature_names = [c for c in data.columns if c not in ("id", "diagnosis")]

            return {
                "model": model,
                "scaler": scaler,
                "data": data,
                "feature_names": feature_names
            }

        resources = load_model_data()
        if resources is None:
            st.stop()

        self.model = resources["model"]
        self.scaler = resources["scaler"]
        self.data = resources["data"]
        self.feature_names = resources["feature_names"]

        # Prepare dataset and compute a test split for metrics
        df = self.data.copy()
        if "diagnosis" not in df.columns:
            st.error("Dataset must contain a 'diagnosis' column with M/B labels.")
            st.stop()

        y = df["diagnosis"].map({"M": 1, "B": 0})
        X = df.drop(columns=[c for c in ["id", "diagnosis"] if c in df.columns], errors="ignore")

        # Train/test split for evaluation in the app
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Fit scaler if not provided
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            # scaler may have been fit on original training data; still transform
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

        # Compute predictions for evaluation
        try:
            y_pred = self.model.predict(X_test_scaled)
            if hasattr(self.model, "predict_proba"):
                y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = None
        except Exception as e:
            st.error(f"Error when using model to predict on test set: {e}")
            st.stop()

        # Store evaluation artifacts as attributes
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.conf_matrix = confusion_matrix(y_test, y_pred)
        self.classification_report_text = classification_report(y_test, y_pred)
        self.accuracy = accuracy_score(y_test, y_pred)

    def create_gauge_chart(self, value, title):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#2ECC71" if value > 0.5 else "#E74C3C"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(231, 76, 60, 0.2)'},
                    {'range': [50, 100], 'color': 'rgba(46, 204, 113, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value * 100
                }
            }
        ))
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def advanced_prediction_visualization(self, prediction, prediction_proba):
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # FIXED: compute confidence correctly based on predicted class
            pred_class = "Malignant" if int(prediction[0]) == 1 else "Benign"
            if prediction_proba is not None:
                # prediction_proba is expected to be an array-like of the Malignant probability
                if int(prediction[0]) == 1:
                    confidence = float(prediction_proba[0])  # P(Malignant)
                else:
                    confidence = float(1 - prediction_proba[0])  # P(Benign)
            else:
                confidence = 0.0

            st.markdown(f"### Prediction: {pred_class}")
            gauge_fig = self.create_gauge_chart(
                confidence,
                f"Confidence Level for {pred_class}"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            if confidence > 0.8:
                st.success("🎯 High Confidence Prediction")
                st.info("💡 Recommended Action: Schedule follow-up with specialist")
            elif confidence > 0.6:
                st.warning("⚠️ Moderate Confidence Prediction")
                st.info("💡 Recommended Action: Additional testing suggested")
            else:
                st.error("⚠️ Low Confidence Prediction")
                st.info("💡 Recommended Action: Comprehensive medical assessment required")

        with col2:
            st.markdown("### Probability Distribution")
            if prediction_proba is not None:
                # FIXED: make sure pie matches predicted class probabilities
                malignant_prob = float(prediction_proba[0])
                benign_prob = 1 - malignant_prob

                prob_data = {
                    'Class': ['Malignant', 'Benign'],
                    'Probability': [malignant_prob, benign_prob]
                }
                fig = go.Figure(data=[go.Pie(
                    labels=prob_data['Class'],
                    values=prob_data['Probability'],
                    hole=.7,
                    textinfo='label+percent',
                    textfont_size=14,
                    hovertemplate="<b>%{label}</b><br>Probability: %{percent}<extra></extra>"
                )])
                fig.update_layout(
                    showlegend=False,
                    annotations=[dict(
                        text='Prediction<br>Confidence',
                        x=0.5, y=0.5,
                        font_size=14,
                        showarrow=False
                    )],
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Probability scores not available for this model.")
        st.markdown('</div>', unsafe_allow_html=True)

    def performance_metrics(self):
        st.header("🔍 Model Performance Analysis")
        tabs = st.tabs([
            "📈 ROC Curve",
            "📊 Precision-Recall",
            "🎯 Confusion Matrix",
            "📑 Classification Report"
        ])

        with tabs[0]:
            self.roc_curve_visualization()
            st.markdown("""
            **Understanding ROC Curve:**
            - Plots True Positive Rate vs False Positive Rate
            - AUC closer to 1.0 indicates better model performance
            - Diagonal line represents random prediction
            """)

        with tabs[1]:
            self.precision_recall_curve()
            st.markdown("""
            **Understanding Precision-Recall Curve:**
            - Shows trade-off between precision and recall
            - Higher curve indicates better model performance
            - Useful for imbalanced classification problems
            """)

        with tabs[2]:
            self.confusion_matrix_heatmap()
            st.markdown("""
            **Understanding Confusion Matrix:**
            - True Positives: Correctly identified positive cases
            - True Negatives: Correctly identified negative cases
            - False Positives: Incorrectly identified positive cases
            - False Negatives: Incorrectly identified positive cases
            """)

        with tabs[3]:
            st.markdown("### Detailed Classification Metrics")
            st.code(self.classification_report_text, language='text')
            st.markdown("""
            **Key Metrics Explained:**
            - Precision: Ratio of correct positive predictions
            - Recall: Ratio of actual positives correctly identified
            - F1-Score: Harmonic mean of precision and recall
            - Support: Number of samples for each class
            """)

    def roc_curve_visualization(self):
        if self.y_pred_proba is None:
            st.info("ROC curve requires probability outputs from the model.")
            return
        y_test = np.array(self.y_test)
        y_pred_proba = np.array(self.y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.3f})',
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgb(52, 152, 219)', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            line=dict(color='rgb(189, 195, 199)', width=2, dash='dash')
        ))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            hovermode='x unified',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    def precision_recall_curve(self):
        if self.y_pred_proba is None:
            st.info("Precision-Recall curve requires probability outputs from the model.")
            return
        y_test = np.array(self.y_test)
        y_pred_proba = np.array(self.y_pred_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_vals, y=precision_vals,
            name='Precision-Recall curve',
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(color='rgb(46, 204, 113)', width=2)
        ))
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            hovermode='x unified',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    def confusion_matrix_heatmap(self):
        conf_matrix = self.conf_matrix
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Benign', 'Predicted Malignant'],
            y=['Actual Benign', 'Actual Malignant'],
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=False
        ))
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    def interactive_feature_selection(self):
        features = {}
        for feature in self.feature_names:
            # safe min/max/mean
            col = self.data[feature]
            min_val = float(col.min())
            max_val = float(col.max())
            mean_val = float(col.mean())
            features[feature] = st.sidebar.slider(
                feature, min_val, max_val, mean_val,
                help=f"Range: {min_val:.2f} - {max_val:.2f} | Mean: {mean_val:.2f}"
            )
        return pd.DataFrame(features, index=[0])

    def run(self):
        st.markdown("<div class='main-header'><h1>🩺 AI Breast Cancer Diagnostic Assistant</h1></div>", unsafe_allow_html=True)
        cols = st.columns(3)
        metrics = [
            ("🎯 Model Accuracy", f"{self.accuracy:.2%}", "Overall predictive performance of the model"),
            ("📊 Total Samples", len(self.data), "Size of the dataset"),
            ("🤖 Model Algorithm", type(self.model).__name__, "Algorithm used")
        ]

        for col, (title, value, help_text) in zip(cols, metrics):
            with col:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric(title, str(value), help=help_text)
                st.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.markdown("## 🔬 Patient Feature Input")
        input_df = self.interactive_feature_selection()
        # scale input safely
        try:
            input_scaled = self.scaler.transform(input_df)
        except Exception:
            # FIXED: don't refit a scaler on a single sample — stop and prompt to retrain
            st.error("❌ Scaler failed to load. Please retrain the model and ensure 'Weight files/scaler.pkl' exists.")
            st.stop()

        prediction = self.model.predict(input_scaled)
        prediction_proba = self.model.predict_proba(input_scaled)[:, 1] if hasattr(self.model, "predict_proba") else None

        st.header("🔬 Diagnostic Prediction")
        # pass the probability array (shape: (n_samples,)) directly
        self.advanced_prediction_visualization(prediction, prediction_proba)

        self.performance_metrics()

        st.markdown("---")
        st.markdown("### ⚕️ Medical Disclaimer")
        st.warning("""
        **Important Notice:**
        - This AI tool is designed for screening purposes only
        - Not a replacement for professional medical diagnosis
        - All results should be interpreted by qualified healthcare professionals
        - Always consult with medical experts for proper diagnosis and treatment
        """)

        st.markdown("---")
        st.markdown("### 👨‍💻 Developer Information")

        # Create two columns for developer information
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("🧑🏻‍💻 Naveen S")
            st.markdown("📞 Contact: +91 7448863062")
            st.markdown("📧 Email: snaveen8105@gmail.com")
            st.markdown("Github: [Click here!](https://github.com/naveeen0308/Breast-cancer-detection)")
            st.markdown("Linkedin: [Click here!](https://www.linkedin.com/in/naveen-s-a70854268/)")

        with col2:
            st.markdown("🧑🏻‍💻 B.Krishna Raja Sree")
            st.markdown("📧 Email: 22b01a4609@svecw.edu.in")
            st.markdown("Github: [Click here!](https://github.com/krishnasree76/)")
            st.markdown("Linkedin: [Click here!](https://www.linkedin.com/in/krishna-raja-sree-bonam-7b6079257/)")
            
        with col3:
            st.markdown("🧑🏻‍💻 Joseph Boban")
            st.markdown("📧 Email: joseph.dm254031@greatlakes.edu.in")
            st.markdown("Github: [Click here!](https://github.com/josephboban2000)")
            st.markdown("Linkedin: [Click here!](https://www.linkedin.com/in/josephboban/)")
         
        with col4:
            st.markdown("🧑🏻‍💻 Shaik Ayesha Parveen")
            st.markdown("📧 Email: ayeshparveen25@gmail.com")
            st.markdown("Github: [Click here!](https://github.com/ShaikAyeshaparveen25/)")
        with col5:
            st.markdown("🧑🏻‍💻 Gayathri R")
            st.markdown("📧 Email: gayathri.22ad@kct.ac.in")
            st.markdown("Github: [Click here!](https://github.com/Gayathri-R-04/)")    


def main():
    app = BreastCancerDetectionApp()
    app.run()


if __name__ == "__main__":
    main()
