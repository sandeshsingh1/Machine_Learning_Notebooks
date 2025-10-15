import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    layout="wide", 
    page_title="🤖 Universal AutoML", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Universal AutoML - Automatic ML pipeline builder"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    .feature-pill {
        background: #e7f3ff;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
        color: #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Universal AutoML Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your CSV, sit back, and watch AI find the best model for your data</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    uploaded = st.file_uploader("📁 Upload CSV File", type=["csv"], help="Upload your dataset in CSV format")
    sample = st.checkbox("🎯 Use Demo Dataset (Titanic)", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.markdown("""
    - ✅ Auto preprocessing
    - 🎯 Smart target detection
    - 🤖 Multiple ML models
    - 📈 Visual comparisons
    - 💾 Download results
    """)

if uploaded is None and not sample:
    # Landing page
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h3>🚀 Quick Start</h3>
            <p>Upload your CSV file and let AutoML handle the rest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>🎯 Auto Detection</h3>
            <p>Automatically detects problem type and target column</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h3>📊 Compare Models</h3>
            <p>Get performance metrics for multiple algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👆 Upload a CSV file from the sidebar or try the demo dataset to get started")
    st.stop()

# Load data
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        data_source = uploaded.name
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        st.stop()
else:
    df = pd.DataFrame({
        "PassengerId":[1,2,3,4,5,6,7,8,9,10],
        "Survived":[0,1,1,1,0,0,0,1,1,0],
        "Pclass":[3,1,3,1,3,3,1,3,2,3],
        "Name":["A","B","C","D","E","F","G","H","I","J"],
        "Sex":["male","female","female","female","male","male","male","female","female","male"],
        "Age":[22,38,26,35,35,np.nan,54,2,27,14],
        "SibSp":[1,1,0,1,0,0,1,1,0,0],
        "Parch":[0,0,0,0,0,0,0,1,0,0],
        "Ticket":["A/5 21171","PC 17599","STON/02. 3101282","113803","373450","17463","34567","PC 17757","36864","A/5 21172"],
        "Fare":[7.25,71.2833,7.925,53.1,8.05,8.4583,51.8625,21.075,11.1333,30.0708],
        "Cabin":[np.nan,"C85",np.nan,"C123",np.nan,np.nan,"C85",np.nan,np.nan,np.nan],
        "Embarked":["S","C","S","S","S","S","C","S","S","C"]
    })
    data_source = "Titanic Demo"

# Dataset overview
st.markdown("### 📋 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0;">📊 Rows</h4>
        <h2 style="margin:0.5rem 0 0 0;">{df.shape[0]:,}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0;">📈 Columns</h4>
        <h2 style="margin:0.5rem 0 0 0;">{df.shape[1]}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0;">❓ Missing</h4>
        <h2 style="margin:0.5rem 0 0 0;">{missing_pct:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0;">💾 Source</h4>
        <h2 style="margin:0.5rem 0 0 0; font-size:1.2rem;">{data_source[:15]}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Data preview
with st.expander("👁️ View Dataset Preview", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)

# Smart target detection
possible_targets = ['target', 'output', 'label', 'y', 'class', 'survived', 'strength', 'price', 'score', 'result']
detected_target = None
for col in df.columns:
    if col.lower() in possible_targets:
        detected_target = col
        break
if detected_target is None:
    detected_target = df.columns[-1]

with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 Target Selection")
    target_col = st.selectbox(
        "Select Target Column", 
        options=list(df.columns), 
        index=list(df.columns).index(detected_target),
        help="The column you want to predict"
    )
    st.caption(f"✨ Auto-detected: {detected_target}")

# Prepare X,y
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# Feature engineering
drop_text_cols = ['name', 'ticket', 'cabin', 'remarks', 'description', 'comments']
X = X.drop(columns=[c for c in X.columns if c.lower() in drop_text_cols], errors='ignore')
id_like_cols = [c for c in X.columns if 'id' in c.lower()]
X = X.drop(columns=id_like_cols, errors='ignore')
X = X.loc[:, X.nunique() > 1]

# Handle missing targets
nan_target_idx = y[y.isna()].index
if len(nan_target_idx) > 0:
    st.warning(f"⚠️ Dropping {len(nan_target_idx)} rows with missing target values")
    X = X.drop(index=nan_target_idx)
    y = y.drop(index=nan_target_idx)

# Detect feature types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Feature info
st.markdown("### 🔍 Feature Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📊 Numeric Features**")
    if num_cols:
        for col in num_cols:
            st.markdown(f'<span class="feature-pill">{col}</span>', unsafe_allow_html=True)
    else:
        st.write("None detected")

with col2:
    st.markdown("**🏷️ Categorical Features**")
    if cat_cols:
        for col in cat_cols:
            st.markdown(f'<span class="feature-pill">{col}</span>', unsafe_allow_html=True)
    else:
        st.write("None detected")

# Problem detection
if y.nunique() <= 15 and y.dtype in ['int64', 'float64']:
    problem_type = "classification"
    prob_icon = "🎯"
else:
    problem_type = "regression"
    prob_icon = "📈"

st.markdown(f"""
<div class="success-box">
    <h3>{prob_icon} Detected Problem Type: <strong>{problem_type.upper()}</strong></h3>
    <p>Target has {y.nunique()} unique values</p>
</div>
""", unsafe_allow_html=True)

# Build preprocessor
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols), 
    ("cat", cat_pipeline, cat_cols)
])

# Model list
if problem_type == "regression":
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Support Vector Regressor": SVR()
    }
else:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', C=1)
    }

# Train button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run AutoML Pipeline"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🤖 Training models..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        results = []
        
        for idx, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}... ({idx+1}/{len(models)})")
            progress_bar.progress((idx + 1) / len(models))
            
            try:
                pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                
                if problem_type == "regression":
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    results.append({
                        "Model": name, 
                        "R²": round(r2, 4), 
                        "MAE": round(mae, 4), 
                        "RMSE": round(rmse, 4)
                    })
                else:
                    acc = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
                    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
                    results.append({
                        "Model": name, 
                        "Accuracy": round(acc, 4), 
                        "Precision": round(prec, 4), 
                        "Recall": round(rec, 4), 
                        "F1": round(f1, 4)
                    })
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {str(e)[:50]}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if len(results) == 0:
            st.error("❌ All models failed. Check your dataset for issues.")
        else:
            results_df = pd.DataFrame(results)
            metric = "R²" if problem_type == "regression" else "Accuracy"
            
            if metric in results_df.columns:
                results_df = results_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
            
            # Results section
            st.markdown("---")
            st.markdown("## 🏆 Results")
            
            # Best model highlight
            best = results_df.iloc[0]
            st.markdown(f"""
            <div class="success-box">
                <h2 style="margin:0;">🥇 Best Model: {best['Model']}</h2>
                <p style="margin:0.5rem 0 0 0;">Performance: {best[metric]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Results table
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📊 Model Comparison")
                st.dataframe(
                    results_df.style.background_gradient(cmap='RdYlGn', subset=[metric]),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("### 📈 Best Model Metrics")
                for col in results_df.columns[1:]:
                    st.metric(col, best[col])
            
            # Visualization
            st.markdown("### 📈 Performance Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_df)))
            
            bars = ax.barh(results_df['Model'], results_df[metric], color=colors)
            ax.set_xlabel(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Model', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Comparison Across Models', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.4f}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download section
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    data=csv,
                    file_name="automl_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                report = f"""AutoML Report
{'='*50}
Dataset: {data_source}
Target Column: {target_col}
Problem Type: {problem_type.upper()}
Samples: {len(X)}
Features: {len(X.columns)}

Best Model: {best['Model']}
Performance ({metric}): {best[metric]}

{'='*50}
All Results:
{results_df.to_string(index=False)}
"""
                st.download_button(
                    "📄 Download Report (TXT)",
                    data=report,
                    file_name="automl_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d;">
    <p>Built with ❤️ using Streamlit | Universal AutoML Studio</p>
    <p style="font-size: 0.9rem;">⚠️ For production use, add cross-validation, hyperparameter tuning, and feature engineering</p>
</div>
""", unsafe_allow_html=True)