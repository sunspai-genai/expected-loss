import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import data
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor




# Set Streamlit layout
st.set_page_config(layout="wide")

# --- Sidebar ---
st.sidebar.title("Prediction of Expected Loss - ML Approach Demo")
selected_section = st.sidebar.radio("Section", [
    "Use Case Approach", "Business Hypothesis", "Exploratory Data Analysis", 
    "Model Development & Evaluation", "Explainability & Governance", "Automated ML Pipeline"
])
###########################################***************************************************************************************
# --- Section: Problem Statement ---
if selected_section == "Use Case Approach":
    st.title("Define Problem Statement")
    st.markdown("""
     
    The objective is to build a **predictive model** that estimates **expected loss** at contract level using historical and contextual features:
    
    - Lease contract attributes  
    - Equipment specifications  
    - Customer profile, past payment behaviour and credit score  
    - Usage and service history  
    - Macroeconomic factors
    """)
    st.divider()
    st.markdown("""          
    Expected loss is defined as the **financial loss** incurred by the leasing company due to equipment failure, service issues, or other factors.
    The model will help in **risk assessment** and **pricing strategies** for leasing contracts.
              
    Target Variable: Expected Loss
                
    **Expected Loss** =  Probability of Default (PD) * Loss Given Default (LGD) * Exposure at Default (EAD)
    - **Probability of Default (PD):** Likelihood that a borrower will default on a loan or lease.
    - **Loss Given Default (LGD):** Percentage of the total exposure that is lost when a borrower defaults.
    - **Exposure at Default (EAD):** Total value at risk at the time of default.
    """)
    st.divider()
    st.markdown("""
    **Machine Learning Approach:**
                
    **Probability of Default (PD)**
    - **Objective:** Predict the probability of default for the Customer or contract within a certain time horizon.
    - **Dataset:** All contracts (defaulted and non-defaulted)
    - **Features:** Customer profile, credit score, payment history, macroeconomic factors.
    - **Model:** Binary Classification.
    - **Evaluation:** Precision, Recall, F1-Score, Confusion Matrix.
    
                
    **Loss Given Default (LGD)**
    - **Objective:** Predict the percentage of loss given default for a specific contract.
    - **Dataset:** Only defaulted contracts
    - **Features:** Equipment type, age, usage history, service incidents.
    - **Model:** Regression.
    - **Evaluation:** Mean Squared Error (MSE), R-squared (R²).
                
    **Exposure at Default (EAD)**
    - **Objective:** Predict the exposure at default for a specific contract.
    - **Dataset:** All contracts (defaulted and non-defaulted)
    - **Features:** Contract value, lease term, payment history.
    - **Model:** Regression.
    - **Evaluation:** Mean Squared Error (MSE), R-squared (R²).
    
    """)
    st.divider()
    image6 = Image.open("images/tdsp-lifecycle2.png")
    st.image(image6, caption="Data Science Life Cycle", use_container_width=False)
    st.markdown("""
    **Data Science Life Cycle**: The process of data science involves several key steps, including data collection, data cleaning, exploratory data analysis, feature engineering, model development, model evaluation, and model deployment. Each step is crucial for building a successful predictive model.
    The data science life cycle is an iterative process, meaning that you may need to revisit previous steps as you gain new insights or as the data changes. This flexibility allows for continuous improvement and adaptation to new challenges.
    The data science life cycle is a collaborative effort that often involves multiple stakeholders, including data scientists, domain experts, and business leaders. Effective communication and collaboration are essential for ensuring that the model meets the needs of the organization and delivers value.""")

    st.divider()
    st.subheader("Traditional Approach vs Machine Learning Approach")

    image7 = Image.open("images/MLMethod/mls3_0101.png")
    st.image(image7, caption="Traditional Approach", use_container_width=False)
    
    st.divider()
    #col1, col2 = st.columns(2)
    #with col1:
    st.subheader("Machine Learning Approach - Level 1")
    image7 = Image.open("images/MLMethod/ml.png")
    st.image(image7, caption="Machine Learning Approach", use_container_width=False)
    #with col2:
    st.subheader("Automated ML Approach")
    image8 = Image.open("images/MLMethod/ml3.png")
    st.image(image8, caption="Automated Approach", use_container_width=False)

    st.divider()
    st.subheader("Model Development Process")
    image9 = Image.open("images/modeldevstep.png")
    st.image(image9, caption="Model Development Process", use_container_width=True)

    st.divider()
    st.subheader("Machine Learning Algorithms")
    image10 = Image.open("images/model.png")
    st.image(image10, caption="Machine Learning Algorithms", use_container_width=True)

    st.divider()
    st.subheader("Model Comparison")
    image11 = Image.open("images/algoadv.png")
    st.image(image11, caption="Model Comparison", use_container_width=True)

    st.divider()
    st.subheader("Model Evaluation")
    image11 = Image.open("images/eval.png")
    st.image(image11, caption="Model Evaluation Metric", use_container_width=True)

    st.divider()
    st.subheader("Python Packages")
    image11 = Image.open("images/pythonpackasges.png")
    st.image(image11, caption="Python Packages", use_container_width=True)

    st.divider()
    st.markdown("""           
   **Points to Consider**
    - The dataset is synthetic and generated to simulate real-world scenarios. It includes various features that are relevant to the leasing industry.
    - The model is trained on historical data and validated using a separate test set. The goal is to achieve a high level of accuracy in predicting expected loss.
    - The model is evaluated using Statistical metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)**.
    - The model is explainable, allowing us to understand the key drivers of expected loss. This is crucial for building trust with stakeholders and ensuring that the model is used effectively in decision-making.
    - The model can be deployed in a production environment, where it can be used to predict expected loss for new contracts. This will help the leasing company to make informed decisions about pricing, risk management, and customer engagement.
    - The model can be continuously monitored and updated as new data becomes available. This will ensure that the model remains accurate and relevant over time.
    - The model can be integrated into the existing systems, allowing for seamless data flow and real-time predictions. This will enable the company to respond quickly to changing market conditions and customer needs.
    - The model can be used to generate reports and dashboards that provide insights into expected loss trends and patterns. This will help the leasing company to identify areas for improvement and optimize its operations.
    
 
    This demo walks through the key steps of data exploration, modeling, and explainability.
    """)
 
###########################################***************************************************************************************
# --- Section: Data Used ---
elif selected_section == "Business Hypothesis":
    st.title("Business Hypothesis & Data Dictionary")
    st.markdown("""
    The dataset includes the following feature categories:
    
    - **Contract Features:** Start date, duration, lease type, region  
    - **Equipment Features:** Type, cost, age at lease, maintenance history  
    - **Customer Profile:** Industry, credit score, segment  
    - **Usage/Service Logs:** Total usage hours, number of breakdowns, downtime  
    - **Historical Losses:** Actual loss from past contracts  
    - **External Data:** Regional economic growth, inflation
    

    
    **Sample Data Preview**:
    """)
    st.subheader("Data Dictionary - PD Model")
    df_variables = pd.DataFrame(data.pd_variables)
    st.table(df_variables)

    st.subheader("Data Dictionary - LGD Model")
    df_variables = pd.DataFrame(data.lgd_variables)
    st.table(df_variables)
    st.subheader("Data Dictionary - EAD Model")
    df_variables = pd.DataFrame(data.ead_variables)
    st.table(df_variables)
    # Simulated sample data

###########################################***************************************************************************************
# --- Section: Statistical Analysis ---
elif selected_section == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("""
    Exploratory Data Analysis (EDA) is a crucial step in the data science process. It involves analyzing and visualizing the data to understand its structure, patterns, and relationships. EDA helps identify potential issues, such as missing values, outliers, and correlations between features. This information is essential for feature selection, data preprocessing, and model development.
    The EDA process typically includes the following steps:
    - **Data Cleaning**: Handling missing values, duplicates, and inconsistencies in the data.
    - **Univariate Analysis**: Analyzing individual features to understand their distributions and characteristics.
    - **Bivariate Analysis**: Exploring relationships between pairs of features, including target variables.
    - **Correlation Analysis**: Identifying correlations between features to understand their relationships and potential multicollinearity.
    - **Feature Engineering**: Creating new features based on existing ones to improve model performance.
    - **Data Visualization**: Using plots and charts to visualize the data and communicate findings effectively.
    EDA is an iterative process, and insights gained during this phase can lead to adjustments in data preprocessing, feature selection, and model development. The goal is to gain a deep understanding of the data and its underlying patterns, which will ultimately inform the modeling process.
    
    """)


    # Set style
    #sns.set(style="whitegrid", palette="muted")
    #st.set_page_config(page_title="EL Model EDA", layout="wide")


    @st.cache_data
    def load_data():
        pd_data = pd.read_csv("data/pd_dataset_v2.csv")
        lgd_data = pd.read_csv("data/lgd_dataset_v2.csv")
        ead_data = pd.read_csv("data/ead_dataset_v2.csv")
        return pd_data, lgd_data, ead_data

    def is_categorical(series):
        return series.dtype == 'object' or series.nunique() <= 10

    def univariate_analysis(df):
        st.subheader("📊 Univariate Analysis")
        feature = st.selectbox("Select a feature", df.columns, key="uni_feature")
        fig, ax = plt.subplots()
        
        if is_categorical(df[feature]):
            st.subheader(f"Count Plot - {feature}")
            sns.countplot(y=feature, data=df, order=df[feature].value_counts().index, ax=ax)
        else:
            st.subheader(f"Histogram & Boxplot - {feature}")
            sns.histplot(df[feature], kde=True, ax=ax)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[feature], ax=ax2)
            st.pyplot(fig2)
            return
        
        st.pyplot(fig)

    def bivariate_analysis(df):
        st.subheader("📈 Bivariate Analysis")
        target = st.selectbox("Select Target Variable", df.columns, key="bi_target")
        feature = st.selectbox("Select Feature Variable", [col for col in df.columns if col != target], key="bi_feature")

        fig, ax = plt.subplots()

        target_is_cat = is_categorical(df[target])
        feature_is_cat = is_categorical(df[feature])

        if not target_is_cat and not feature_is_cat:
            st.subheader(f"Scatter Plot: {feature} vs {target}")
            sns.scatterplot(x=feature, y=target, data=df, ax=ax, alpha=0.6)
            st.pyplot(fig)

            corr = df[[feature, target]].corr().iloc[0, 1]
            st.markdown(f"**Correlation coefficient:** {corr:.2f}")

        elif target_is_cat and not feature_is_cat:
            st.subheader(f"Box Plot: {feature} by {target}")
            sns.boxplot(x=target, y=feature, data=df, ax=ax)
            st.pyplot(fig)

        elif not target_is_cat and feature_is_cat:
            st.subheader(f"Box Plot: {target} by {feature}")
            sns.boxplot(x=feature, y=target, data=df, ax=ax)
            st.pyplot(fig)

        else:
            st.subheader(f"Cross Tab Heatmap: {feature} vs {target}")
            ctab = pd.crosstab(df[feature], df[target], normalize='index')
            sns.heatmap(ctab, annot=True, cmap="YlGnBu", ax=ax)
            st.pyplot(fig)

    def feature_importance(df, target):
        st.header("🔥 Feature Importance")
        X = df.drop(columns=[target])
        y = df[target]

        # Drop non-numeric/categorical columns from X
        X = pd.get_dummies(X, drop_first=True)

        if is_categorical(y):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x=importances.values[:20], y=importances.index[:20], ax=ax)
        ax.set_title(f"Top Feature Importances for {target}")
        st.pyplot(fig)

    def correlation_heatmap(df):
        st.header("🧭 Correlation Heatmap (Numerical Features Only)")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)

    # Load data
    pd_data, lgd_data, ead_data = load_data()

    # Sidebar navigation
    st.sidebar.subheader("Select Model Dataset")
    model_choice = st.sidebar.radio("Choose:", ["Probability of Default", "Loss Given Default", "Exposure at Default"])

    if model_choice == "Probability of Default":
        st.subheader("📌 PD Model EDA (Probability of Default)")
        df = pd_data

    elif model_choice == "Loss Given Default":
        st.subheader("📌 LGD Model EDA (Loss Given Default)")
        df = lgd_data

    elif model_choice == "Exposure at Default":
        st.subheader("📌 EAD Model EDA (Exposure at Default)")
        df = ead_data

    # EDA sections
    univariate_analysis(df)
    bivariate_analysis(df)
    feature_importance(df, target=st.selectbox("Select Target for Feature Importance", df.columns, key="fi_target"))
    correlation_heatmap(df)

################################################################
# @st.cache_data
# def load_data():
#     pd_data = pd.read_csv("data/pd_dataset_v2.csv")
#     lgd_data = pd.read_csv("data/lgd_dataset_v2.csv")
#     ead_data = pd.read_csv("data/ead_dataset_v2.csv")
#     return pd_data, lgd_data, ead_data

# def univariate_analysis(df):
#     st.header("📊 Univariate Analysis")
    
#     # Dropdown to select a feature for univariate analysis
#     feature = st.selectbox("Select a feature for Univariate Analysis", df.columns)
    
#     st.subheader(f"Univariate Analysis for Feature: {feature}")
    
#     fig, ax = plt.subplots()
#     if df[feature].dtype in ['int64', 'float64']:
#         # Histogram
#         sns.histplot(df[feature], kde=True, ax=ax)
#         st.pyplot(fig)

#         # Boxplot
#         fig2, ax2 = plt.subplots()
#         sns.boxplot(x=df[feature], ax=ax2)
#         st.pyplot(fig2)
#     else:
#         # Countplot for categorical features
#         fig, ax = plt.subplots()
#         sns.countplot(y=feature, data=df, order=df[feature].value_counts().index, ax=ax)
#         st.pyplot(fig)

# def bivariate_plot(df, target):
#     # Dropdown to select a feature for bivariate analysis
#     features = [col for col in df.columns if col != target]
#     selected_feature = st.selectbox(f"Select feature for Bivariate Analysis (target: {target})", features)
    
#     st.subheader(f"Bivariate Analysis: {selected_feature} vs {target}")
    
#     fig, ax = plt.subplots()
#     if df[target].nunique() <= 2:
#         sns.boxplot(x=target, y=selected_feature, data=df, ax=ax)
#     else:
#         sns.scatterplot(x=selected_feature, y=target, data=df, alpha=0.5, ax=ax)
#     st.pyplot(fig)

# def feature_importance(df, target, model_type='classification'):
#     st.subheader("🔥 Feature Importance")
#     X = df.drop(columns=[target])
#     y = df[target]

#     if model_type == "classification":
#         model = RandomForestClassifier(n_estimators=100, random_state=42)
#     else:
#         model = RandomForestRegressor(n_estimators=100, random_state=42)
    
#     model.fit(X, y)
#     importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

#     fig, ax = plt.subplots()
#     sns.barplot(x=importances.values, y=importances.index, ax=ax)
#     ax.set_title(f"Feature Importance for {target}")
#     st.pyplot(fig)

# def correlation_heatmap(df, title):
#     st.subheader("🧭 Correlation Heatmap")
#     corr = df.corr()
#     fig, ax = plt.subplots()
#     sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
#     ax.set_title(title)
#     st.pyplot(fig)

# # Load datasets
# pd_data, lgd_data, ead_data = load_data()

# # Sidebar
# st.sidebar.title("Target Variable Selector")
# model_choice = st.sidebar.radio("Choose:", ["Probability of Default", "Loss Given Default", "Exposure at Default"])

# # Routing based on model selection
# if model_choice == "Probability of Default":
#     st.title("📌 PD Model EDA (Probability of Default)")
    
#     # Univariate Analysis for PD data
#     univariate_analysis(pd_data)
    
#     # Bivariate Analysis for PD data
#     bivariate_plot(pd_data, target="Default")
    
#     # Feature Importance & Correlation Heatmap
#     feature_importance(pd_data, target="Default", model_type="classification")
#     correlation_heatmap(pd_data, title="PD Feature Correlation")

# elif model_choice == "Loss Given Default":
#     st.title("📌 LGD Model EDA (Loss Given Default)")
    
#     # Univariate Analysis for LGD data
#     univariate_analysis(lgd_data)
    
#     # Bivariate Analysis for LGD data
#     bivariate_plot(lgd_data, target="LGD")
    
#     # Feature Importance & Correlation Heatmap
#     feature_importance(lgd_data, target="LGD", model_type="regression")
#     correlation_heatmap(lgd_data, title="LGD Feature Correlation")

# elif model_choice == "Exposure at Default":
#     st.title("📌 EAD Model EDA (Exposure at Default)")
    
#     # Univariate Analysis for EAD data
#     univariate_analysis(ead_data)
    
#     # Bivariate Analysis for EAD data
#     bivariate_plot(ead_data, target="EAD")
    
#     # Feature Importance & Correlation Heatmap
#     feature_importance(ead_data, target="EAD", model_type="regression")
#     correlation_heatmap(ead_data, title="EAD Feature Correlation")

#####################################new code##########################################
    # @st.cache_data
    # def load_data():
    #     pd_data = pd.read_csv("pd_dataset_v2.csv")
    #     lgd_data = pd.read_csv("lgd_dataset_v2.csv")
    #     ead_data = pd.read_csv("ead_dataset_v2.csv")
    #     return pd_data, lgd_data, ead_data

    # def bivariate_plot(df, target):
    #     features = [col for col in df.columns if col != target and df[col].dtype != 'object']
    #     st.subheader("📊 Bivariate Plots")
    #     selected_feature = st.selectbox("Select feature", features)
        
    #     fig, ax = plt.subplots()
    #     if df[target].nunique() <= 2:
    #         sns.boxplot(x=target, y=selected_feature, data=df, ax=ax)
    #     else:
    #         sns.scatterplot(x=selected_feature, y=target, data=df, alpha=0.5, ax=ax)
    #     st.pyplot(fig)

    # def feature_importance(df, target, model_type='classification'):
    #     st.subheader("🔥 Feature Importance")
    #     X = df.drop(columns=[target])
    #     y = df[target]

    #     if model_type == "classification":
    #         model = RandomForestClassifier(n_estimators=100, random_state=42)
    #     else:
    #         model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    #     model.fit(X, y)
    #     importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    #     fig, ax = plt.subplots()
    #     sns.barplot(x=importances.values, y=importances.index, ax=ax)
    #     ax.set_title(f"Feature Importance for {target}")
    #     st.pyplot(fig)

    # def correlation_heatmap(df, title):
    #     st.subheader("🧭 Correlation Heatmap")
    #     corr = df.corr()
    #     fig, ax = plt.subplots()
    #     sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
    #     ax.set_title(title)
    #     st.pyplot(fig)

    # # Load datasets
    # pd_data, lgd_data, ead_data = load_data()

    # # Sidebar
    # st.sidebar.title("Target Variable Selector")
    # model_choice = st.sidebar.radio("Choose:", ["Probability of Default", "Loss Given Default", "Exposure at Default"])

    # # Routing based on model selection
    # if model_choice == "Probability of Default":
    #     st.title("📌 PD Model EDA (Probability of Default)")
    #     bivariate_plot(pd_data, target="Default")
    #     feature_importance(pd_data, target="Default", model_type="classification")
    #     correlation_heatmap(pd_data, title="PD Feature Correlation")

    # elif model_choice == "Loss Given Default":
    #     st.title("📌 LGD Model EDA (Loss Given Default)")
    #     bivariate_plot(lgd_data, target="LGD")
    #     feature_importance(lgd_data, target="LGD", model_type="regression")
    #     correlation_heatmap(lgd_data, title="LGD Feature Correlation")

    # elif model_choice == "Exposure at Default":
    #     st.title("📌 EAD Model EDA (Exposure at Default)")
    #     bivariate_plot(ead_data, target="EAD")
    #     feature_importance(ead_data, target="EAD", model_type="regression")
    #     correlation_heatmap(ead_data, title="EAD Feature Correlation")


###########################################***************************************************************************************
# --- Section: Model Development ---
elif selected_section == "Model Development & Evaluation":
    st.title("Model Development & Evaluation")
    st.subheader("Steps involved in Model Development")
    st.markdown("""
    - **Feature Selection**: Identify and select relevant features for the model. ( Chi-square test, correlation analysis, ANOVA, Recursive Feature Elimination (RFE), forward feature selection, backward feature selection, LASSO regression, decision trees)
    - **Data Preprocessing**: Clean and preprocess the data, including handling missing values, encoding categorical variables, and scaling numerical features.
    - **Handling imbalanced data**: Use techniques like SMOTE or ADASYN to balance the dataset.
    - **Train-Test Split**: Split the dataset into training and testing sets to evaluate model performance.
    - **Model Selection**: Choose appropriate machine learning algorithms for the task.
    - **Model Training**: Train the model using the training dataset.
    - **Model Evaluation**: Evaluate the model's performance using various metrics such as accuracy, precision, recall, and F1-score.
    - **Model Tuning**: Fine-tune the model parameters to improve performance.
    - **Model Deployment**: Deploy the model for batch predictions.
    - **Model Monitoring**: Monitor the model's performance over time and update it as necessary.
    """)
    st.subheader("Train Test Split")
    image = Image.open("images/traintest.png")
    st.image(image, caption="Model Evaluation Metrics", use_container_width=False)

    st.sidebar.title("Select Model Dataset")
    model_choice = st.sidebar.radio("Choose:", ["Probability of Default", "Loss Given Default", "Exposure at Default"])
    st.divider()
    if model_choice == "Probability of Default":
    
            st.subheader("Probability of Default model - Model Evaluation")
            st.markdown("""
                        - **Train Test Split**: 70% for training and 30% for testing.
                        - **Model Selection**: XGBoost Classifier.
                        - **Output**: Probability Score (between 0-1) of Default (PD) for each contract.
                        - **Evaluation Metrics**:
                            - **Accuracy**: Overall correctness of the model.
                            - **Precision**: Correctly predicted positive observations to the total predicted positives.
                            - **Recall**: Correctly predicted positive observations to all actual positives.
                            - **F1-Score**: Weighted average of Precision and Recall.
                            - **Support**: Number of actual occurrences of the class in the specified dataset.
                        """)


        
            data_rows = []
            for label, metrics in data.report_dict.items():
                if isinstance(metrics, dict):
                    row = {'class': label}
                    row.update(metrics)
                    data_rows.append(row)
                else:
                    # Scalar value (accuracy)
                    data_rows.append({'class': 'accuracy', 'precision': metrics, 'recall': '', 'f1-score': '', 'support': ''})

            df_report = pd.DataFrame(data_rows)

            # Round values
            for col in ['precision', 'recall', 'f1-score']:
                df_report[col] = pd.to_numeric(df_report[col], errors='coerce').round(2)

            df_report['support'] = pd.to_numeric(df_report['support'], errors='coerce').fillna('').astype(str)

            # --- Convert summary metrics to DataFrame ---
            df_summary = pd.DataFrame(list(data.summary_metrics.items()), columns=["Metric", "Value"])
            df_summary["Value"] = df_summary["Value"].round(2)

            # --- Display in Streamlit ---
            st.subheader("Model Evaluation Summary")

            st.subheader("🔹 Key Performance Metrics")
            st.dataframe(df_summary, use_container_width=False)

            st.subheader("🔸 Detailed Classification Report")
            st.dataframe(df_report)
            st.markdown("""- **Evaluation Metrics**:
                            - **Accuracy**: Overall correctness of the model.
                            - **Precision**: Correctly predicted positive observations to the total predicted positives.
                            - **Recall**: Correctly predicted positive observations to all actual positives.
                            - **F1-Score**: Weighted average of Precision and Recall.
                            - **Support**: Number of actual occurrences of the class in the specified dataset.
                        """)

            st.divider()
            st.image("images/pd/output.png", caption="Confusion Matrix & ROC Curve", use_container_width=True)
            st.markdown("""
            - **Confusion Matrix**: This matrix shows the number of correct and incorrect predictions made by the model.
            - **ROC Curve**: The ROC curve illustrates the trade-off between sensitivity and specificity for every possible cut-off. The area under the ROC curve (AUC) is a measure of the model's ability to distinguish between positive and negative classes.""")
        
    if model_choice == "Loss Given Default": 
            st.subheader("Loss Given Default model - Model Evaluation")
            st.markdown("""
                        - **Train Test Split**: 70% for training and 30% for testing.
                        - **Model Selection**: XGBoost Regressor.
                        - **Output**: Expected Loss (LGD) for each contract.
                        - **Evaluation Metrics**:
                            - **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.
                            - **R-squared (R²)**: Indicates how well the independent variables explain the variability of the dependent variable.
                            - **Actual vs Predicted**: A plot showing the relationship between actual and predicted values.
                            - **Residual Distribution**: A plot showing the distribution of residuals (errors).
                            - **Feature Importance**: A plot showing the importance of each feature in predicting the target variable.
                        """)
            st.subheader("LGD Model - Model Evaluation")
            image = Image.open("images/lgd/output.png")
            st.image(image, caption="Actual vs Predicted", use_container_width=False)
            st.image("images/lgd/lgd2.png", caption="Residual Distribution", use_container_width=False)
            st.image("images/lgd/lgd3.png", caption="Feature Importance", use_container_width=False)

    if model_choice == "Exposure at Default":
        st.subheader("Exposure at Default model - Model Evaluation")
        st.markdown("""
                    - **Train Test Split**: 70% for training and 30% for testing.
                    - **Model Selection**: XGBoost Regressor.
                    - **Output**: Expected Loss (EAD) for each contract.
                    - **Evaluation Metrics**:
                        - **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.
                        - **R-squared (R²)**: Indicates how well the independent variables explain the variability of the dependent variable.
                        - **Actual vs Predicted**: A plot showing the relationship between actual and predicted values.
                        - **Residual Distribution**: A plot showing the distribution of residuals (errors).
                        - **Feature Importance**: A plot showing the importance of each feature in predicting the target variable.
                    """)
        st.subheader("EAD Model - Model Evaluation")
        image = Image.open("images/ead/ead1.png")
        st.image(image, caption="Actual vs Predicted", use_container_width=False)
        st.image("images/ead/ead2.png", caption="Residual Distribution", use_container_width=False)
        st.image("images/ead/ead3.png", caption="Feature Importance", use_container_width=False)
        
###########################################***************************************************************************************
# --- Section: Model Explainability ---
elif selected_section == "Explainability & Governance":
     st.title("Model Interpretability, Explainability and Governance")
     st.markdown("""
     Model Interpretability and explainability are crucial for understanding the model's predictions, regulatory compliance and building trust with stakeholders.
     They are essential for ensuring that the model is used effectively in decision-making, and for identifying areas for improvement.""" )
     st.markdown("""
                 **Why interpretability and explainability matter**
                 - **Interpretability** refers to the degree to which a human can understand the decision that’s been made by a model.
                 - **Explainability** goes further by providing insights into the specific contribution of each feature to the final prediction.
                 """)
     st.markdown("""
                 **In our scenario we will apply below techniques for Explainability and Interpretability**
                - **ELI5 (Explain like I am 5)**: This library provides a simple way to visualize and understand the model's predictions. It can be used to explain individual predictions or the overall model behavior.
                - **Feature Importance**: This technique ranks the features based on their contribution to the model's predictions. It helps identify which features are most influential in predicting expected loss.
                - **SHAP (SHapley Additive exPlanations)**: It assigns each feature an importance value for a particular prediction, allowing us to understand the impact of each feature on the model's output.
                - **LIME (Local Interpretable Model-agnostic Explanations)**: It provides local explanations for individual predictions, helping to understand why the model made a specific prediction.
                - **Partial Dependence Plots (PDP)**: These plots show the relationship between a feature and the predicted outcome, while averaging out the effects of other features.
            
                 """)
     
     st.sidebar.subheader("Select Model Dataset")
     model_choice = st.sidebar.radio("Choose:", ["Probability of Default", "Loss Given Default", "Exposure at Default"])

     if model_choice == "Probability of Default":
        st.subheader("📌 PD Model EDA (Probability of Default)")
        st.subheader("ELI5 (Explain like I am 5)")
        image = Image.open("images/pd/pd_eli5.jpeg")
        st.image(image, caption="ELI5 Feature Importance", use_container_width=False)
        st.markdown("""
                    - In this table, the Weight column gives the importance score associated with each feature 
                    - while the Feature column displays the name of the input variable that contributes to the models predictions. 
                    - A higher weight indicates that the feature has a larger influence on the predictions made by the model. 
                    - The plus/minus symbol represents the uncertainty or standard deviation around the weight estimate. 
                    - It gives an indication of the variability in the feature importance across different instances.""")

        st.divider()
        st.subheader("SHAP (SHapley Additive exPlanations) - Global")
        image = Image.open("images/pd/pd_shap.png")
        st.image(image, caption="SHAP Global", use_container_width=False)
        st.markdown("""
                    - **SHAP**: This technique assigns each feature an importance value for a particular prediction, allowing us to understand the impact of each feature on the model's output.
                    - The SHAP plot shows the distribution of SHAP values for each feature across all instances in the dataset.
                    - The features are ranked by their importance, with the most influential features appearing at the top.
                    - The color of the dots indicates whether the feature value is high (red) or low (blue) for that instance.
                    - The SHAP values indicate the direction and magnitude of the feature's impact on the model's prediction.
                    - The SHAP summary plot provides a global view of feature importance and the direction of their impact on the model's predictions.""")
        
        st.divider()

        # col1, col2 = st.columns(2)
        # with col1:
        #     st.subheader("SHAP - Local 1")
        #     image = Image.open("images/pd/local1shap.png")
        #     st.image(image, caption="Example 1", use_column_width=True)
        # with col2:    
        #     st.subheader("SHAP - Local 2")
        #     image = Image.open("images/pd/local2shap.png")
        #     st.image(image, caption="Example 2", use_column_width=True)

        st.subheader("LIME (Local Interpretable Model-agnostic Explanations) - Local")
        # image = Image.open("images/pd/pd_lime.png")
        # st.image(image, caption="LIME", use_column_width=True)
        with open('images/lime_explanation.html', "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, width=800)
        st.markdown("""
                    - **LIME**: This technique provides local explanations for individual predictions, helping to understand why the model made a specific prediction.
                    - The LIME plot shows the contribution of each feature to the model's prediction for a specific instance. 
                    - The features are ranked by their importance, with the most influential features appearing at the top.""")
        st.divider()
        st.subheader("Partial Dependence Plots (PDP)")
        image = Image.open("images/pd/pd_pdp.png")
        st.image(image, caption="PDP", use_container_width=True)
        st.markdown("""
                    - **Partial Dependence Plots (PDP)**: These plots show the relationship between a feature and the predicted outcome, while averaging out the effects of other features.
                    - The PDP plot illustrates how the predicted probability of default changes with different values of the selected feature, while keeping other features constant.
                    - This helps to understand the marginal effect of a feature on the model's predictions.""")
     


        
        # st.subheader("Feature Importance")
        # image = Image.open("images/pd/featureimp.jpg")
        # st.image(image, caption="Feature Importance", use_column_width=True)
        
        # st.markdown("""
        #             - **Feature Importance**: This technique ranks the features based on their contribution to the model's predictions.""")

     elif model_choice == "Loss Given Default":
        st.subheader("📌 LGD Model EDA (Loss Given Default)")
        

     elif model_choice == "Exposure at Default":
        st.subheader("📌 EAD Model EDA (Exposure at Default)")
            


elif selected_section == "Automated ML Pipeline":
    st.subheader("ML Pipeline")
    image12 = Image.open("images/A3F9AD6E-FB51-4400-B3E9-6C1FF164C0EE.jpeg")
    st.image(image12, caption="ML Pipeline", use_container_width=True)
    #st.subheader("ML Pipeline")
    st.divider()
    image5 = Image.open("images/pipe.jpeg")
    st.image(image5, caption="Automated ML Pipeline", use_container_width=True)


