import streamlit as st
import pickle 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

df = pd.read_csv(r"C:\Users\acer\Downloads\capestone_main\survey.csv")
st.set_page_config(page_title="Divyans Capestone [24127015]", layout="wide")

# Session state for page tracking
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Navigation: only if NOT on Home
if st.session_state.page != "Home":
    page = st.sidebar.radio(
        "Go to",
        ["Home", "EDA", "Classification Model", "Regression Model", "Unsupervised Model"],
        index=["Home", "EDA", "Classification Model", "Regression Model", "Unsupervised Model"].index(st.session_state.page)
    )
    st.session_state.page = page

# Home page content
if st.session_state.page == "Home":
    st.header(":blue[CAPESTONE PROJECT OPEN LEARN 1.0, AUG 2025]",divider='blue')
    st.header(":blue[ABOUT THIS PROJECT]",divider='blue')
    st.markdown("""
    ## HELLO
    Greetings to everyone reading this. This website is created to showcase my Capstone project, which I completed on *14 August 2025*.

    In this website, a project related to various machine learning models built on many different machine learning algorithms, 
    like ** Logistic regression*,PCA(principal component analysis),k-Means Clustering* and many more, with multi insights along with EDA.
    """)
    st.header(":blue[NOTE]",divider='blue')
    st.markdown(""" 
    ##REFER  MY GITHUB REPO
    *Refer for all notebooks each meant for EDA,SUPERVISED...etc,.  
    """)
    st.header(":blue[ABOUT THE DATASET]",divider='blue')
    st.markdown("""
    ## DATASET
    This dataset is taken from the ‘2014 Mental Health in Tech Survey’ conducted by ‘Open Sourcing Mental Illness (OSMI)’:  
    The purpose of this dataset is to evaluate working class people's mental health status.  
    Dataset initially contained a total 1259 rows along with 23 columns.
    """)
    features = [
        "Age", "Gender", "Country", "self_employed", "family_history",
        "treatment", "work_interfere", "no_employees", "remote_work",
        "tech_company", "benefits", "care_options", "wellness_program",
        "seek_help", "anonymity", "leave", "mental_health_consequence",
        "phys_health_consequence", "coworkers", "supervisor",
        "mental_health_interview", "phys_health_interview", 
        "mental_vs_physical", "obs_consequence"
    ]
    st.table({"value": features})

    # # Button to go to EDA
    # if st.button("Go to EDA"):
    #     st.session_state.page = "EDA"
    #     st.rerun()
st.write("This dataset is taken from the ‘2014 Mental Health in Tech Survey’ conducted by ‘Open Sourcing Mental Illness (OSMI)’. The purpose of this dataset is to evaluate working class people’s mental health status. dataset initially contained a total 1259 rows along with 27 columns")
st.write('various features of this dataset are')
st.dataframe(df.columns.to_list())

st.header(":blue[ABOUT OPEN LEARN COHORT]",divider='blue')
st.write('[Open learn cohort](https://www.openlearn.org.in/) was organised at Dr. B. R. Ambedkar National Institute of Technology, Jalandhar in year of 2025 Aimed to bring an AI/ML revolution in the college. Open cohort is led by key industry figures and pioneers are mentored by experts in AI/ML domain')