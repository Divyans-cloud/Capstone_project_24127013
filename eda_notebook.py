# # %%
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # %% [markdown]
# # # Welcome to the EDA stuff (Exploratory Data Analysis) and Stuff Bro..
# # 

# # %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

# print("Path to dataset files:", path)

# # %% [markdown]
# # # Import Libraries

# # %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Optional To See all Columns and Set Styles
# sns.set(style="whitegrid")
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)

# # %% [markdown]
# # # Load the DataSets
# # # Head Structure

# # %%
# # Load the datasets
# df = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')# mh stands for mental health dataset
# # Quick look
# df.head() # first five rows


# # %% [markdown]
# # # tail structure

# # %%
# df.tail() # last five rows

# # %% [markdown]
# # # Understand the Data¶

# # %%
# print("Mental Health Shape:",df.shape)

# # info
# df.info
# df.describe(include='all')
# # NULL VALUES
# print("Mental Health  nulls:\n",df.isnull().sum())

# # %% [markdown]
# # # Step 4.Data Cleaning

# # %%
# # Load, drop unnecessary columns
# df_cleaned = df.drop(columns=['Timestamp','state','comments'])
# # # Fill Nulls
# # matches_cleaned['city']=matches_cleaned['city'].fillna("Unknown")
# # matches_cleaned['winner']=matches_cleaned['winner'].fillna("Draw")
# #Check Duplicates
# print("Duplicates in mental health:",df_cleaned.duplicated().sum())
# print(df_cleaned .head(),'\n')
# print(df_cleaned.tail())


# # %% [markdown]
# # # REAL HERO EDA...

# # %%
# # Cleaning the data

# df= df[(df['Age'] >= 18) & (df['Age'] <= 65)].copy()


# def clean_gender(g):
#     g = str(g).strip().lower()
#     male_vals = ['male', 'm', 'man', 'cis male', 'cis man', 'male (cis)', 'malr', 'msle', 'maile', 'mal', 'make', 'mail', 'guy (-ish) ^_^', 'male-ish', 'something kinda male?', 'ostensibly male, unsure what that really means']
#     female_vals = ['female', 'f', 'woman', 'cis female', 'femail', 'femake', 'female (cis)', 'cis-female/femme', 'female (trans)', 'trans-female', 'trans woman']
#     if g in male_vals:
#         return 'Male'
#     elif g in female_vals:
#         return 'Female'
#     else:
#         return 'Other'

# df['Gender'] = df['Gender'].apply(clean_gender)



# def clean_self_employed(val):
#     val = str(val).strip().lower()
#     if val in ['yes', 'y']:
#         return 'Yes'
#     elif val in ['no', 'n']:
#         return 'No'
#     else:
#         return np.nan
# df['self_employed'] = df['self_employed'].apply(clean_self_employed)
# df['self_employed'].fillna('No', inplace=True)


# def clean_work_interfere(val):
#     if pd.isna(val):
#         return np.nan
#     val = str(val).strip().lower()
#     if val in ['never']:
#         return 'Never'
#     elif val in ['rarely']:
#         return 'Rarely'
#     elif val in ['sometimes']:
#         return 'Sometimes'
#     elif val in ['often']:
#         return 'Often'
#     else:
#         return np.nan
# df['work_interfere'] = df['work_interfere'].apply(clean_work_interfere)
# df['work_interfere'].fillna("Don't know", inplace=True)

# # %%
# df['no_employees'] = df['no_employees'].str.strip()
# employee_mapping = {
#     "1-5": "1-5",
#     "6-25": "6-25",
#     "26-100": "26-100",
#     "100-500": "100-500",
#     "500-1000": "500-1000",
#     "More than 1000": "1000+",
#     "1000+": "1000+"
# }
# df['no_employees'] = df['no_employees'].map(employee_mapping).fillna(df['no_employees'])

# cols_to_clean = [
#     'mental_health_consequence',
#     'phys_health_consequence',
#     'coworkers',
#     'mental_health_interview',
#     'phys_health_interview',
#     'mental_vs_physical',
#     'obs_consequence'
# ]
# for col in cols_to_clean:
#     df[col] = df[col].astype(str).str.strip().str.lower().str.title()


# # %%
# df['comments'] = df['comments'].astype(str).str.strip()
# df['comments'] = df['comments'].replace('nan', np.nan)


# print(df.head(10))  


# # %% [markdown]
# # # Mental Health Treatment by Gender¶
# # Mental Health Treatment by Gender Shows gender-based trends:
# # Do males, females, or others seek treatment more often?
# # 
# # Is there any gender bias or stigma?

# # %%
# #Mental Health Treatment by Gender
# plt.figure(figsize=(12, 6))
# sns.countplot(data=df, x='Gender', hue='treatment', palette='Set2')
# plt.title('Mental Health Treatment by Gender', fontsize=16)
# plt.xlabel('Gender', fontsize=14)
# plt.ylabel('Number of Respondents', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(title='Sought Treatment', fontsize=12, title_fontsize=13)
# plt.grid(axis='y', linestyle='--')
# plt.tight_layout()
# plt.show()


# # %% [markdown]
# # # Mental Health Treatment by Country

# # %% [markdown]
# # # 📊 Why These Plots Matter for Project:
# # 
# # 1.Mental Health Treatment by Country
# # 2.KEY ANLAYSIS Shows geographical trends: Are people from some countries more likely to seek          treatment?
# # Help in analysis
# # 
# # Country-wise awareness/support
# # 
# # Cultural differences around mental health

# # %%
# #Mental Health Treatment by Country
# top_countries = df['Country'].value_counts().head(10).index
# plt.figure(figsize=(12, 6))
# sns.countplot(data=df[df['Country'].isin(top_countries)], y='Country', hue='treatment', palette='coolwarm')
# plt.title('Mental Health Treatment by Country (Top 10)')
# plt.xlabel('Count')
# plt.ylabel('Country')
# plt.legend(title='Sought Treatment')
# plt.tight_layout()
# plt.show()

# # %% [markdown]
# # 📌 1. Mental Health Treatment by Age Group 🔍 What This Plot Shows: This plot helps us understand how age affects mental health treatment behavior. Are younger employees more willing to seek help? Do older individuals hesitate to take treatment?
# # 
# # ❓ Questions This Plot Answers:
# # 
# # 1.Which age group seeks treatment most often?
# # 
# #  a)Is there a visible trend across age ranges?
# # 2.Do mental health concerns vary with age?# 

# # %%
# # First, create age groups
# bins = [18, 25, 35, 45, 55, 65]
# labels = ['18–25', '26–35', '36–45', '46–55', '56–65']
# df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)

# # Plot
# plt.figure(figsize=(8,5))
# sns.countplot(data=df, x='age_group', hue='treatment', palette='Set2')
# plt.title('Mental Health Treatment by Age Group')
# plt.xlabel('Age Group')
# plt.ylabel('Number of People')
# plt.legend(title='Sought Treatment')
# plt.tight_layout()
# plt.show()

# # %% [markdown]
# # # Mental Health Treatment by Work Interference¶
# # 📌 Mental Health Treatment by Work Interference 🔍 What This Plot Shows: This visual shows how much mental health issues interfere with people’s work and how that connects to seeking treatment.
# # 
# # ❓ Questions This Plot Answers:
# # 
# #   1. Do people who report more interference (e.g., "Often") also seek more help?
# # 
# #  2. Are people ignoring their mental health even if it affects their work?
# # 
# #  3.Can workplace productivity signal mental health needs?

# # %%
# plt.figure(figsize=(7,4))
# sns.countplot(data=df, x='work_interfere', hue='treatment', palette='Set2')
# plt.title('Mental Health Treatment vs. Work Interference')
# plt.xlabel('Work Interference Level')
# plt.ylabel('Number of People')
# plt.legend(title='Sought Treatment')
# plt.tight_layout()
# plt.show()

# # %% [markdown]
# # # Impact of Company Benefits on Mental Health Treatment

# # %% [markdown]
# # # 📌 3. Impact of Company Benefits on Treatment 🔍 What This Plot Shows: This plot explores if having mental health benefits at the workplace (like therapy support or insurance) makes people more likely to seek treatment.
# # 
# # ❓ Questions This Plot Answers:
# # 
# # 1.Are employees with access to benefits taking better care of their mental health?
# # 
# # 2.Does lack of support stop people from getting treatment?
# # 
# # 3.Is the company's attitude toward mental health making a difference?

# # %%
# plt.figure(figsize=(7,5))
# sns.countplot(data=df, x='benefits', hue='treatment', palette='coolwarm')
# plt.title('Impact of Company Mental Health Benefits on Treatment')
# plt.xlabel('Has Mental Health Benefits')
# plt.ylabel('Number of People')
# plt.legend(title='Sought Treatment')
# plt.tight_layout()
# plt.show()


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import kagglehub


    # In[3]:
import seaborn
import numpy as  sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import os


def run():
        
    # file_path = os.path.join(os.path.dirname(_file_), "survey.csv")
    df = pd.read_csv(r"C:\Users\acer\Downloads\capestone_main\survey.csv")

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

    # # Welcome to the EDA stuff (Exploratory Data Analysis) and  Stuff Bro..

    # In[2]:


    import kagglehub


    # In[3]:

    #import seaborn as sns

    # Optional To See all Columns and Set Styles
    # sns.set(style="whitegrid")
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_colwidth', None)


    # # Load the DataSets

    # # Head Structure

    # In[4]:


    # Load the datasets
    # df = pd.read_csv('survey.csv')# mh stands for mental health dataset
    # Quick look
    df.head() # first five rows


    # # tail structure

    # In[5]:


    df.tail() # last five rows

    # # Understand the Data

    # In[6]:


    print("Mental Health Shape:",df.shape)

    # info
    df.info
    df.describe(include='all')
    # NULL VALUES
    print("Mental Health  nulls:\n",df.isnull().sum())

    # # Step 4.Data Cleaning

    # In[7]:


    # Load, drop unnecessary columns
    df_cleaned = df.drop(columns=['Timestamp','state','comments'])
    # # Fill Nulls
    # matches_cleaned['city']=matches_cleaned['city'].fillna("Unknown")
    # matches_cleaned['winner']=matches_cleaned['winner'].fillna("Draw")
    #Check Duplicates
    print("Duplicates in mental health:",df_cleaned.duplicated().sum())

    print("shape after dropping above columns:",df_cleaned.shape)

    # # REAL HERO EDA...

    # In[8]:


    # Cleaning the data

    df= df[(df['Age'] >= 18) & (df['Age'] <= 65)].copy()


    def clean_gender(g):
        g = str(g).strip().lower()
        male_vals = ['male', 'm', 'man', 'cis male', 'cis man', 'male (cis)', 'malr', 'msle', 'maile', 'mal', 'make', 'mail', 'guy (-ish) ^_^', 'male-ish', 'something kinda male?', 'ostensibly male, unsure what that really means']
        female_vals = ['female', 'f', 'woman', 'cis female', 'femail', 'femake', 'female (cis)', 'cis-female/femme', 'female (trans)', 'trans-female', 'trans woman']
        if g in male_vals:
            return 'Male'
        elif g in female_vals:
            return 'Female'
        else:
            return 'Other'

    df['Gender'] = df['Gender'].apply(clean_gender)



    def clean_self_employed(val):
        val = str(val).strip().lower()
        if val in ['yes', 'y']:
            return 'Yes'
        elif val in ['no', 'n']:
            return 'No'
        else:
            return np.nan
    df['self_employed'] = df['self_employed'].apply(clean_self_employed)
    df['self_employed'].fillna('No', inplace=True)

    def clean_work_interfere(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if val in ['never']:
            return 'Never'
        elif val in ['rarely']:
            return 'Rarely'
        elif val in ['sometimes']:
            return 'Sometimes'
        elif val in ['often']:
            return 'Often'
        else:
            return np.nan
    df['work_interfere'] = df['work_interfere'].apply(clean_work_interfere)
    df['work_interfere'].fillna("Don't know", inplace=True)



    # In[9]:


    df['no_employees'] = df['no_employees'].str.strip()
    employee_mapping = {
        "1-5": "1-5",
        "6-25": "6-25",
        "26-100": "26-100",
        "100-500": "100-500",
        "500-1000": "500-1000",
        "More than 1000": "1000+",
        "1000+": "1000+"
    }
    df['no_employees'] = df['no_employees'].map(employee_mapping).fillna(df['no_employees'])

    cols_to_clean = [
        'mental_health_consequence',
        'phys_health_consequence',
        'coworkers',
        'mental_health_interview',
        'phys_health_interview',
        'mental_vs_physical',
        'obs_consequence'
    ]
    for col in cols_to_clean:
        df[col] = df[col].astype(str).str.strip().str.lower().str.title()

    # In[10]:


    df['comments'] = df['comments'].astype(str).str.strip()
    df['comments'] = df['comments'].replace('nan', np.nan)


    print(df.head(10))  

    # # Mental Health Treatment by Gender

    # 1. Mental Health Treatment by Gender
    # Shows gender-based trends:
    # 
    # Do males, females, or others seek treatment more often?
    # 
    # Is there any gender bias or stigma?

    # In[11]:


    #Mental Health Treatment by Gender
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Gender', hue='treatment', palette='Set2')
    plt.title('Mental Health Treatment by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Number of Respondents', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Sought Treatment', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    # 

    # # Mental Health Treatment by Country

    # 📊 Why These Plots Matter for Project:
    # 
    #     1.Mental Health Treatment by Country
    #     2.KEY ANLAYSIS Shows geographical trends: Are people from some countries more likely to seek          treatment?
    # Help in analysis
    # 
    # Country-wise awareness/support
    # 
    # Cultural differences around mental health

    # In[12]:


    #Mental Health Treatment by Country
    top_countries = df['Country'].value_counts().head(10).index
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df[df['Country'].isin(top_countries)], y='Country', hue='treatment', palette='coolwarm')
    plt.title('Mental Health Treatment by Country (Top 10)')
    plt.xlabel('Count')
    plt.ylabel('Country')
    plt.legend(title='Sought Treatment')
    plt.tight_layout()
    plt.show()

    # # Mental Health Treatment by Age Group

    # 📌 1. Mental Health Treatment by Age Group
    # 🔍 What This Plot Shows:
    # This plot helps us understand how age affects mental health treatment behavior.
    # Are younger employees more willing to seek help?
    # Do older individuals hesitate to take treatment?
    # 
    # ❓ Questions This Plot Answers:
    # 
    #    1. Which age group seeks treatment most often?
    #      
    #     2. Is there a visible trend across age ranges?
    # 
    #    3. Do mental health concerns vary with age?

    # In[13]:


    # First, create age groups
    bins = [18, 25, 35, 45, 55, 65]
    labels = ['18–25', '26–35', '36–45', '46–55', '56–65']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # Plot
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='age_group', hue='treatment', palette='Set2')
    plt.title('Mental Health Treatment by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of People')
    plt.legend(title='Sought Treatment')
    plt.tight_layout()
    plt.show()

    # #   Mental Health Treatment by Work Interference

    # 📌  Mental Health Treatment by Work Interference
    # 🔍 What This Plot Shows:
    # This visual shows how much mental health issues interfere with people’s work and how that connects to seeking treatment.
    # 
    # ❓ Questions This Plot Answers:
    # 
    #       1. Do people who report more interference (e.g., "Often") also seek more help?
    # 
    #      2. Are people ignoring their mental health even if it affects their work?
    # 
    #      3.Can workplace productivity signal mental health needs?

    # In[14]:


    plt.figure(figsize=(7,4))
    sns.countplot(data=df, x='work_interfere', hue='treatment', palette='Set2')
    plt.title('Mental Health Treatment vs. Work Interference')
    plt.xlabel('Work Interference Level')
    plt.ylabel('Number of People')
    plt.legend(title='Sought Treatment')
    plt.tight_layout()
    plt.show()

    # #  Impact of Company Benefits on  Mental Health Treatment

    # 📌 3. Impact of Company Benefits on Treatment
    # 🔍 What This Plot Shows:
    # This plot explores if having mental health benefits at the workplace (like therapy support or insurance) makes people more likely to seek treatment.
    # 
    # ❓ Questions This Plot Answers:
    # 
    #     1.Are employees with access to benefits taking better care of their mental health?
    # 
    #     2.Does lack of support stop people from getting treatment?
    # 
    #     3.Is the company's attitude toward mental health making a difference?

    # In[15]:


    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x='benefits', hue='treatment', palette='coolwarm')
    plt.title('Impact of Company Mental Health Benefits on Treatment')
    plt.xlabel('Has Mental Health Benefits')
    plt.ylabel('Number of People')
    plt.legend(title='Sought Treatment')
    plt.tight_layout()
    plt.show()

    # In[ ]: