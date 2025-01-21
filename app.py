import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve

# Custom CSS for RTL, Vazirmatn font, and styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@100..900&display=swap');
        
        * {
            font-family: 'Vazirmatn', sans-serif;
            # direction: rtl;
            # text-align: right;
        }
        
        h2, h3, h4, h5, h6 {
            font-family: 'Vazirmatn', sans-serif;
            direction: rtl;
            text-align: right;
        } 

        .stSidebar {
            direction: rtl;
            text-align: right;
            }    

        .stButton button {
            width: 100%;
        }
        
        .stSelectbox, .stNumberInput {
            direction: rtl;
            text-align: right;
        }
        
        .stTitle, .stHeader, .stSubheader, h1 {
            font-family: 'Vazirmatn', sans-serif;
            text-align: center;
            direction: ltr;
        }
        
        .result {
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 20px;
        }
        
        .probability {
            color: #1f77b4;
        }
        
        .qualitative {
            color: #ff7f0e;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.title("DelayGuard")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('Capital_Project_Schedules_and_Budgets.xlsx')
    df.columns = df.columns.str.strip()
    date_columns = ['Project Phase Actual Start Date', 'Project Phase Planned End Date', 'Project Phase Actual End Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%m/%d/%Y', errors='coerce')
    df['Delay'] = (df['Project Phase Actual End Date'] > df['Project Phase Planned End Date']).astype(int)
    df = df.dropna(subset=['Delay'])
    df['Project Budget Amount'] = pd.to_numeric(df['Project Budget Amount'], errors='coerce')
    return df

df = load_data()

# Feature selection
features = ['Project Geographic District', 'Project Type', 'Project Phase Name', 
            'Project Status Name', 'Project Budget Amount']
X = df[features]
y = df['Delay']

# Preprocessing
categorical_features = ['Project Geographic District', 'Project Type', 'Project Phase Name', 'Project Status Name']
numeric_features = ['Project Budget Amount']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Load or train the model
@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load('decision_tree_model_with_smote.pkl')
    except FileNotFoundError:
        model = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=300,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_resampled, y_train_resampled)
        joblib.dump(model, 'decision_tree_model_with_smote.pkl')
    return model

model = load_or_train_model()

# Define the function to map probabilities to qualitative outputs
def get_qualitative_output(probability):
    if probability < 0.2:
        return "خیلی کم احتمال"
    elif probability < 0.4:
        return "کم احتمال"
    elif probability < 0.6:
        return "خنثی"
    elif probability < 0.8:
        return "محتمل"
    else:
        return "خیلی محتمل"

# Mapping dictionaries
types_mapping = {
    "اسکان موقت آموزشی (سازمان ساخت مدارس)": "SCA IEH",
    "حل‌وفصل تنظیمات عملیاتی مدارس (آموزش‌وپرورش)": "DOE - RESOA",
    "پروژه‌های فنی و حرفه‌ای (آموزش‌وپرورش)": "DOE - Skilled Trades",
    "برنامه بهبود سرمایه‌ای (سازمان ساخت مدارس)": "SCA CIP",
    "برنامه بهبود سرمایه‌ای و تنظیمات عملیاتی مدارس": "SCA CIP RESOA",
    "پروژه‌های فوری و سریع‌الاجرا": "Fast Track Projects",
    "تنظیمات عملیاتی فناوری اطلاعات (آموزش‌وپرورش)": "DIIT - RESOA",
    "واکنش اضطراری (سازمان ساخت مدارس)": "SCA Emergency Response",
    "تجهیزات و مبلمان مدارس (سازمان ساخت مدارس)": "SCA Furniture & Equipment",
    "پروژه‌های فضای عمومی (اعتماد برای زمین عمومی)": "Trust For Public Land",
    "حذف رنگ‌های سرب‌دار (آموزش‌وپرورش)": "DOE - Lead Paint",
    "افزایش ظرفیت مدارس (سازمان ساخت مدارس)": "SCA Capacity",
    "پروژه‌های مهدکودک برای کودکان سه‌ساله": "3K",
    "بهبود سایت‌های اجاره‌ای (سازمان ساخت مدارس)": "SCA Lease Site Improvement",
    "پروژه اتصال فناوری اطلاعات (آموزش‌وپرورش)": "DIIT - Project Connect",
    "نورپردازی اضطراری مدارس (سازمان ساخت مدارس)": "SCA Emergency Lighting",
    "پروژه‌های پیش‌دبستانی": "PRE-K",
    "پیش‌دبستانی تحت مدیریت آموزش‌وپرورش": "DOE Managed PRE-K",
}

phases_mapping = {
    "ساخت‌وساز": "Construction",
    "تعریف پروژه": "Scope",
    "طراحی": "Design",
    "مدیریت ساخت، مبلمان و تجهیزات": "CM,F&E",
    "خرید و نصب": "Purch & Install",
    "مبلمان و تجهیزات": "F&E",
    "مدیریت ساخت، هنر، مبلمان و تجهیزات": "CM,Art,F&E",
    "مدیریت ساخت": "CM",
}

status_mapping = {
    "در حال انجام": "In-Progress",
    "تکمیل شده": "Complete",
    "در انتظار شروع": "PNS",
}

# Sidebar for additional data
st.sidebar.header("اطلاعات بیش‌تر")

# About the App Section
st.sidebar.subheader("درباره برنامه")
st.sidebar.write("""
این برنامه از یک مدل یادگیری ماشین برای پیش‌بینی احتمال تاخیر در پروژه‌های سرمایه‌ای استفاده می‌کند. 
با وارد کردن اطلاعات پروژه، می‌توانید احتمال تاخیر و خروجی کیفی آن را مشاهده کنید.
""")

# Dataset Overview Section
st.sidebar.subheader("بررسی اجمالی داده‌ها")
st.sidebar.write("نمونه‌ای از داده‌های استفاده شده:")
st.sidebar.dataframe(df.sample(5))
st.sidebar.write(f"تعداد رکوردها: {len(df)}")
st.sidebar.write(f"تعداد پروژه‌های با تاخیر: {df['Delay'].sum()}")

# Model Performance Section
st.sidebar.subheader("عملکرد مدل")
y_pred = model.predict(X_test_preprocessed)
y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]

st.sidebar.write("دقت مدل:")
st.sidebar.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

st.sidebar.write("ماتریس اشتباه:")
st.sidebar.write(confusion_matrix(y_test, y_pred))

# st.sidebar.write("گزارش طبقه‌بندی:")
# st.sidebar.write(classification_report(y_test, y_pred))

# GitHub Link
st.sidebar.markdown("---")
st.sidebar.write("Follow me on [Github](https://github.com/okaeiz)")

# Input fields for user data
st.header("لطفا ویژگی‌های ورودی را وارد کنید")

col1, col2 = st.columns(2)

with col1:
    project_district = st.selectbox(
        "منطقه جغرافیایی پروژه",
        options=[f"منطقه {i}" for i in range(1, 33)]
    )

    project_type_labels = st.selectbox(
        "نوع پروژه",
        list(types_mapping.keys())
    )
    project_type = types_mapping[project_type_labels]

with col2:
    project_phase_labels = st.selectbox("فاز پروژه", list(phases_mapping.keys()))
    project_phase = phases_mapping[project_phase_labels]

    project_status_labels = st.selectbox(
        "وضعیت پروژه",
        list(status_mapping.keys())  
    )
    project_status = status_mapping[project_status_labels]

project_budget = st.number_input(
    "مبلغ بودجه پروژه",
    value=0.0
)

# Predict button
if st.button("پیش‌بینی"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Project Geographic District': [project_district],
        'Project Type': [project_type],
        'Project Phase Name': [project_phase],
        'Project Status Name': [project_status],
        'Project Budget Amount': [project_budget]
    })

    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Predict probabilities
    predicted_probabilities = model.predict_proba(input_data_preprocessed)[:, 1]

    # Get qualitative output
    qualitative_output = get_qualitative_output(predicted_probabilities[0])

    # Display the result with custom styling
    st.subheader("نتیجه پیش‌بینی")
    st.markdown(
        f"""
        <div class="result">
            <span class="probability">احتمال تاخیر: {predicted_probabilities[0] * 100:.2f}%</span><br>
            <span class="qualitative">خروجی کیفی: {qualitative_output}</span>
        </div>
        """,
        unsafe_allow_html=True
    )