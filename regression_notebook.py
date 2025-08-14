# # %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

# print("Path to dataset files:", path)

# # %% [markdown]
# # # Step 1.1 Load the Libraries

# # %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.linear_model import Lasso,LogisticRegression,LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestRegressor

# # %% [markdown]
# # # Step 2. Choose Target & Features

# # %%
# df = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')

# # %% [markdown]
# # # Have A look...

# # %%
# df.head() #first five rows

# # %%
# X = [
#     'Gender',
#     'family_history',
#     'treatment',
#     'tech_company',
#     'benefits',
#     'care_options',
#     'wellness_program',
#     'seek_help',
# ]

# y = 'Age'

# # %% [markdown]
# # # Handling missing Data

# # %%
# df = df.dropna(subset=['Age'])
# for col in X:
#     if df[col].dtype == 'object':
#         df[col] = df[col].fillna('Unknown')
#     else:
#         df[col] = df[col].fillna(df[col].median())

# # %% [markdown]
# # # NEW INFO

# # %%
# df.info()
# df.describe()

# # %% [markdown]
# # # Step 3.Features & target split

# # %%
# valid_min, valid_max = 0, 120
# df = df[(df['Age'] >= valid_min) & (df['Age'] <= valid_max)]
# X = df[X]

# y = df['Age']

# # %% [markdown]
# # # Step 4.Encoding categorical data and Some PreProcessing Steps

# # %%
# # Identify categorical & numeric columns
# categorical_cols = X.select_dtypes(include=['object']).columns
# numeric_cols = X.select_dtypes(exclude=['object']).columns

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#     ],
#     remainder='passthrough'
# )

# # %% [markdown]
# # # Step 5. Model Training

# # %%
# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(random_state=42))
# ])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model.fit(X_train, y_train)

# # %%
# print(df['Age'].unique())
# print(df['Age'].dtype)
# print(df['Age'].describe())

# # %% [markdown]
# # # Step6. Finally Verify Metrics

# # %%
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"MAE: {mae:.2f}") # On average, our model’s predicted age is off by about 5–6 years.
# print(f"MSE: {mse:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.2f}")
#!/usr/bin/env python
# coding: utf-8

# In[122]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


import os
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso,LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

    # # Step 2. Choose Target & Features
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # Step 1. Load & See the Data

# In[123]:


import kagglehub
def run():
    # Download latest version
    # file_path = os.path.join(os.path.dirname(_file_), "survey.csv")
    # file_path = os.path.join(os.path.dirname(_file_), "survey", "survey.csv")

    df = pd.read_csv(r"C:\Users\acer\Downloads\capestone_main\survey.csv")

    # # Step 1.1 Load the Libraries

    # In[124]:


   


    df.head() #first five rows

    # In[127]:


    X = [
        'Gender',
        'family_history',
        'treatment',
        'tech_company',
        'benefits',
        'care_options',
        'wellness_program',
        'seek_help',
    ]

    y = 'Age'

    # # Handling missing Data

    # In[128]:


    df = df.dropna(subset=['Age'])
    for col in X:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())


    # # NEW INFO

    # In[129]:


    df.info()
    df.describe()

    # # Step 3.Features & target split

    # In[130]:


    valid_min, valid_max = 0, 120
    df = df[(df['Age'] >= valid_min) & (df['Age'] <= valid_max)]
    X = df[X]

    y = df['Age']

    # # Step 4.Encoding categorical data and Some PreProcessing Steps

    # In[131]:


    # Identify categorical & numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )


    # # Step 5. Model Training

    # In[132]:



    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    import pickle
    with open("regression.pkl", "wb") as f:

        pickle.dump(model, f)  # replace model with your classifier variable name
    print("✅ regression.pkl saved successfully")


    # In[133]:


    print(df['Age'].unique())
    print(df['Age'].dtype)
    print(df['Age'].describe())


    # # Step6. Finally Verify Metrics

    # In[134]:


    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}") # On average, our model’s predicted age is off by about 5–6 years.
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


