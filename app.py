#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

#logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

#session state initialization
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

#folder setup
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, "data", "raw")
clean_dir = os.path.join(base_dir, "data", "cleaned")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

log("application started")
log(f"raw_dir = {raw_dir}")
log(f"clean_dir = {clean_dir}")

#page config
st.set_page_config("End-to-End Naive Bayes", layout="wide")
st.title("End-to-End Naive Bayes Classifier Application")

#sidebar: model settings
st.sidebar.header("Naive Bayes Settings")

nb_type = st.sidebar.selectbox(
    "Naive Bayes Variant",
    ["Gaussian NB", "Multinomial NB", "Bernoulli NB", "Complement NB"]
)

st.sidebar.markdown("### Model Parameters")

if nb_type == "Gaussian NB":
    var_smoothing = st.sidebar.slider("Var Smoothing (log10)", -12, -1, -9)
    var_smoothing = 10 ** var_smoothing
    st.sidebar.info("Gaussian NB: Best for continuous features with normal distribution")
    
elif nb_type == "Multinomial NB":
    alpha = st.sidebar.slider("Alpha (Smoothing)", 0.0, 2.0, 1.0, 0.1)
    fit_prior = st.sidebar.checkbox("Fit Prior Probabilities", value=True)
    st.sidebar.info("Multinomial NB: Best for discrete count features (e.g., text data)")
    
elif nb_type == "Bernoulli NB":
    alpha = st.sidebar.slider("Alpha (Smoothing)", 0.0, 2.0, 1.0, 0.1)
    fit_prior = st.sidebar.checkbox("Fit Prior Probabilities", value=True)
    binarize = st.sidebar.slider("Binarize Threshold", 0.0, 1.0, 0.0, 0.1)
    if binarize == 0.0:
        binarize = None
    st.sidebar.info("Bernoulli NB: Best for binary/boolean features")
    
else:  # Complement NB
    alpha = st.sidebar.slider("Alpha (Smoothing)", 0.0, 2.0, 1.0, 0.1)
    fit_prior = st.sidebar.checkbox("Fit Prior Probabilities", value=True)
    norm = st.sidebar.checkbox("Normalize Weights", value=False)
    st.sidebar.info("Complement NB: Best for imbalanced datasets")

use_scaling = st.sidebar.checkbox("Use Feature Scaling", value=False)
use_grid_search = st.sidebar.checkbox("Use Grid Search CV")

log(f"Naive Bayes settings - type: {nb_type}")

#step 1: Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1: Data Ingestion started")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])
df = None
raw_path = None

if option == "Download Dataset":
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Iris (Classification)", "Penguins (Classification)", "Titanic (Classification)"]
    )
    
    if st.button("Download Dataset"):
        log(f"Downloading {dataset_choice} dataset")
        
        if dataset_choice == "Iris (Classification)":
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            filename = "iris.csv"
        elif dataset_choice == "Penguins (Classification)":
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
            filename = "penguins.csv"
        else:  # Titanic
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
            filename = "titanic.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            raw_path = os.path.join(raw_dir, filename)
            with open(raw_path, "wb") as f:
                f.write(response.content)

            df = pd.read_csv(raw_path)
            st.success(f"{dataset_choice} Dataset Downloaded successfully")
            log(f"{dataset_choice} dataset saved at {raw_path}")
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            log(f"Error downloading dataset: {e}")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(raw_dir, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded file saved at {raw_path}")

#step 2: EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2: EDA started")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Duplicate Rows:**", df.duplicated().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # Class distribution (if we can identify categorical columns)
    st.subheader("Categorical Features Analysis")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        selected_cat = st.selectbox("Select categorical feature to analyze", categorical_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Value Counts for {selected_cat}:**")
            st.write(df[selected_cat].value_counts())
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            df[selected_cat].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {selected_cat}')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, center=0)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap")

    # Box plots for numeric features
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_features = st.multiselect(
            "Select features to visualize",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        
        if selected_features:
            n_cols = 2
            n_rows = (len(selected_features) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for idx, col in enumerate(selected_features):
                if idx < len(axes):
                    axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(True, alpha=0.3)
            
            for idx in range(len(selected_features), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

    log("EDA completed")

#step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")
    log("Step 3: Data Cleaning started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Missing Value Handling Strategy",
            ["Mean", "Median", "Mode", "Drop Rows"]
        )
    
    with col2:
        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
    
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
        log("Dropped rows with missing values")
    else:
        # Handle numeric columns
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().sum() > 0:
                if strategy == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    log(f"Filled missing values in {col} with mean")
                elif strategy == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    log(f"Filled missing values in {col} with median")
                elif strategy == "Mode":
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                    log(f"Filled missing values in {col} with mode")
        
        # Handle categorical columns with mode
        for col in df_clean.select_dtypes(include=['object']).columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                log(f"Filled missing values in {col} with mode")
    
    # Remove duplicates
    if remove_duplicates:
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_len - len(df_clean)
        if removed > 0:
            st.info(f"Removed {removed} duplicate rows")
            log(f"Removed {removed} duplicate rows")
    
    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cleaned Data Preview:**")
        st.dataframe(df_clean.head())
    with col2:
        st.write("**Cleaning Summary:**")
        st.write(f"Original rows: {len(df)}")
        st.write(f"Cleaned rows: {len(df_clean)}")
        st.write(f"Missing values: {df_clean.isnull().sum().sum()}")
    
    log("Data cleaning completed")
else:
    st.info("Please complete Step 1 to proceed.")

#step 4: Save cleaned dataset
st.header("Step 4: Save Cleaned Dataset")
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data to save. Please complete Step 3.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_data_{timestamp}.csv"
        clean_path = os.path.join(clean_dir, clean_filename)

        st.session_state.df_clean.to_csv(clean_path, index=False)
        st.success("Cleaned dataset saved successfully")
        st.info(f"Cleaned dataset saved at {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")
        st.session_state.cleaned_saved = True

#step 5: Load cleaned dataset
st.header("Step 5: Load Cleaned Dataset")
clean_files = os.listdir(clean_dir)
if not clean_files:
    st.warning("No cleaned datasets found. Please save one in Step 4")
    log("No cleaned datasets found")
    df_model = None
else:
    selected = st.selectbox("Select cleaned dataset", clean_files)
    df_model = pd.read_csv(os.path.join(clean_dir, selected))
    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")
    
    st.dataframe(df_model.head())

#step 6: Train Naive Bayes Model
if df_model is not None:
    st.header(f"Step 6: Train {nb_type} Classifier")
    log(f"Step 6: Train {nb_type} started")

    target = st.selectbox("Select target variable", df_model.columns)
    
    if st.button("Train Model"):
        y = df_model[target].copy()
        
        # Encode target if it's categorical
        le_target = None
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            log("Target column encoded")
            st.info(f"Target encoded: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
        
        # Select features (exclude target)
        x = df_model.drop(columns=[target])
        
        # Handle categorical features in X
        categorical_features = x.select_dtypes(include=['object']).columns.tolist()
        if categorical_features:
            st.info(f"Encoding categorical features: {categorical_features}")
            for col in categorical_features:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col].astype(str))
                log(f"Encoded feature: {col}")
        
        # Select only numeric features
        x = x.select_dtypes(include=[np.number])
        
        if x.empty:
            st.error("No numerical features available for training after encoding.")
            st.stop()

        log(f"Features selected: {list(x.columns)}")
        st.write(f"**Features used:** {list(x.columns)}")
        st.write(f"**Number of samples:** {len(x)}")
        st.write(f"**Number of features:** {len(x.columns)}")
        st.write(f"**Number of classes:** {len(np.unique(y))}")

        # Check for class imbalance
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        st.write("**Class Distribution:**", class_dist)
        
        # Visualize class distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(unique, counts, color='skyblue', edgecolor='black')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        st.pyplot(fig)

        # Train-test split
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42, stratify=y
        )
        log(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

        # Feature Scaling (optional but can help some NB variants)
        if use_scaling:
            if nb_type in ["Multinomial NB", "Complement NB"]:
                # Use MinMaxScaler for non-negative features
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            st.info(f"Features scaled using {type(scaler).__name__}")
            log(f"Features scaled using {type(scaler).__name__}")

        # Handle negative values for Multinomial/Complement NB
        if nb_type in ["Multinomial NB", "Complement NB"]:
            if np.any(x_train < 0):
                st.warning("Negative values detected. Applying MinMaxScaler to ensure non-negative features.")
                scaler = MinMaxScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                log("Applied MinMaxScaler due to negative values")

        if use_grid_search:
            st.info("Using Grid Search CV for hyperparameter tuning...")
            log("Using Grid Search CV")
            
            if nb_type == "Gaussian NB":
                param_grid = {'var_smoothing': np.logspace(-12, -1, 20)}
                base_model = GaussianNB()
            elif nb_type == "Multinomial NB":
                param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}
                base_model = MultinomialNB()
            elif nb_type == "Bernoulli NB":
                param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}
                base_model = BernoulliNB()
            else:  # Complement NB
                param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False], 'norm': [True, False]}
                base_model = ComplementNB()
            
            grid = GridSearchCV(
                base_model, 
                param_grid=param_grid, 
                scoring='accuracy',
                cv=5,
                n_jobs=-1
            )
            
            with st.spinner("Training with Grid Search... This may take a while..."):
                grid.fit(x_train, y_train)
            
            st.success("Grid Search completed!")
            st.write("**Best Parameters:**")
            st.json(grid.best_params_)
            st.write(f"**Best Cross-Validation Score:** {grid.best_score_:.4f}")
            
            model = grid.best_estimator_
            log(f"Best params: {grid.best_params_}")
            log(f"Best CV score: {grid.best_score_:.4f}")
        else:
            # Model initialization with user settings
            if nb_type == "Gaussian NB":
                model = GaussianNB(var_smoothing=var_smoothing)
            elif nb_type == "Multinomial NB":
                model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            elif nb_type == "Bernoulli NB":
                model = BernoulliNB(alpha=alpha, fit_prior=fit_prior, binarize=binarize)
            else:  # Complement NB
                model = ComplementNB(alpha=alpha, fit_prior=fit_prior, norm=norm)
            
            model.fit(x_train, y_train)
            st.success(f"{nb_type} model trained successfully")
            log(f"{nb_type} model trained successfully")

        # Predictions
        y_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)
        
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(x_test)

        # Classification Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{train_acc:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{test_acc:.4f}")
        with col3:
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            st.metric("F1 Score", f"{f1:.4f}")
        
        # Check for overfitting
        if train_acc - test_acc > 0.15:
            st.warning("⚠️ Potential overfitting detected!")
        
        log(f"Train Accuracy: {train_acc:.4f}")
        log(f"Test Accuracy: {test_acc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {nb_type}', fontsize=14)
        
        # Add class labels if available
        if le_target is not None:
            ax.set_xticklabels(le_target.classes_, rotation=45)
            ax.set_yticklabels(le_target.classes_, rotation=0)
        
        st.pyplot(fig)
        log("Confusion matrix displayed")

        # Classification Report
        st.subheader("Classification Report")
        if le_target is not None:
            target_names = le_target.classes_
        else:
            target_names = [f"Class {i}" for i in np.unique(y)]
        
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        log("Classification report displayed")

        # Feature Probabilities (for Gaussian NB)
        if nb_type == "Gaussian NB" and hasattr(model, 'theta_'):
            st.subheader("Feature Statistics per Class")
            
            theta_df = pd.DataFrame(
                model.theta_,
                columns=x.columns,
                index=[f"Class {i}" if le_target is None else le_target.classes_[i] for i in range(len(model.theta_))]
            )
            
            st.write("**Mean values per class:**")
            st.dataframe(theta_df)
            
            # Visualize feature means
            fig, ax = plt.subplots(figsize=(12, 6))
            theta_df.T.plot(kind='bar', ax=ax)
            ax.set_title('Feature Means by Class')
            ax.set_xlabel('Features')
            ax.set_ylabel('Mean Value')
            plt.xticks(rotation=45)
            plt.legend(title='Class')
            plt.tight_layout()
            st.pyplot(fig)

        # Prior Probabilities
        st.subheader("Prior Class Probabilities")
        if hasattr(model, 'class_prior_'):
            prior_df = pd.DataFrame({
                'Class': [f"Class {i}" if le_target is None else le_target.classes_[i] for i in range(len(model.class_prior_))],
                'Prior Probability': model.class_prior_
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(prior_df)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(prior_df['Class'], prior_df['Prior Probability'], color='coral', edgecolor='black')
                ax.set_xlabel('Class')
                ax.set_ylabel('Prior Probability')
                ax.set_title('Class Prior Probabilities')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # Cross-validation scores
        st.subheader("Cross-Validation Analysis")
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
        with col2:
            st.metric("Std CV Score", f"{cv_scores.std():.4f}")
        with col3:
            st.metric("Min/Max", f"{cv_scores.min():.4f} / {cv_scores.max():.4f}")
        
        # Plot CV scores
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, 6), cv_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='green')
        ax.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cross-Validation Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        log(f"CV scores: {cv_scores}")

        # Prediction Probability Distribution
        st.subheader("Prediction Confidence Analysis")
        
        # Get max probability for each prediction
        max_proba = np.max(y_pred_proba, axis=1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of confidence
        ax1.hist(max_proba, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Confidence')
        ax1.axvline(x=max_proba.mean(), color='r', linestyle='--', label=f'Mean: {max_proba.mean():.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot of confidence by correctness
        correct = y_pred == y_test
        confidence_data = [max_proba[correct], max_proba[~correct]]
        ax2.boxplot(confidence_data, labels=['Correct', 'Incorrect'])
        ax2.set_ylabel('Prediction Confidence')
        ax2.set_title('Confidence: Correct vs Incorrect Predictions')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(f"**Average confidence for correct predictions:** {max_proba[correct].mean():.4f}")
        st.write(f"**Average confidence for incorrect predictions:** {max_proba[~correct].mean():.4f}")

        # Model Information
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Model Type:** {type(model).__name__}")
            st.write(f"**Number of Classes:** {len(model.classes_)}")
            st.write(f"**Number of Features:** {x.shape[1]}")
        with col2:
            st.write(f"**Training Samples:** {len(x_train)}")
            st.write(f"**Test Samples:** {len(x_test)}")
            if nb_type == "Gaussian NB":
                st.write(f"**Var Smoothing:** {model.var_smoothing:.2e}")

        # Model Comparison Section
        st.subheader("Compare with Other Naive Bayes Variants")
        if st.checkbox("Run comparison with all NB variants"):
            with st.spinner("Training all Naive Bayes variants..."):
                models_to_compare = {
                    'Gaussian NB': GaussianNB(),
                    'Bernoulli NB': BernoulliNB()
                }
                
                # Only add Multinomial and Complement if no negative values
                if np.all(x_train >= 0):
                    models_to_compare['Multinomial NB'] = MultinomialNB()
                    models_to_compare['Complement NB'] = ComplementNB()
                
                comparison_results = []
                
                for name, m in models_to_compare.items():
                    m.fit(x_train, y_train)
                    train_score = m.score(x_train, y_train)
                    test_score = m.score(x_test, y_test)
                    cv_score = cross_val_score(m, x_train, y_train, cv=5, scoring='accuracy').mean()
                    
                    comparison_results.append({
                        'Model': name,
                        'Train Accuracy': train_score,
                        'Test Accuracy': test_score,
                        'CV Score': cv_score
                    })
                
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Test Accuracy', 'CV Score']))
                
                # Visualize comparison
                fig, ax = plt.subplots(figsize=(12, 6))
                x_pos = np.arange(len(comparison_df))
                width = 0.25
                
                ax.bar(x_pos - width, comparison_df['Train Accuracy'], width, label='Train Accuracy', alpha=0.8)
                ax.bar(x_pos, comparison_df['Test Accuracy'], width, label='Test Accuracy', alpha=0.8)
                ax.bar(x_pos + width, comparison_df['CV Score'], width, label='CV Score', alpha=0.8)
                
                ax.set_xlabel('Model')
                ax.set_ylabel('Accuracy')
                ax.set_title('Naive Bayes Variants Comparison')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(comparison_df['Model'], rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
                
                log("Model comparison completed")

        log("Model training and evaluation completed")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit 🎈")
log("Application running")
