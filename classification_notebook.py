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
# # # BACK TO THE CLASSIFICATION TASK

# # %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

# print("Path to dataset files:", path)

# # %% [markdown]
# # # LOAD THE KNIVES ...IMPORT LIBRARIES

# # %%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.linear_model import Lasso,LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import OneHotEncoder

# # %% [markdown]
# # # Load the DataSet 

# # %%
# df = pd.read_csv('/kaggle/input/mental-health-in-tech-survey/survey.csv')

# # %%
# print("Mental Health Shape:",df.shape)

# # info
# df.info
# df.describe(include='all')
# # NULL VALUES
# print("Mental Health  nulls:\n",df.isnull().sum())

# # %% [markdown]
# # # Lets take a Quick look

# # %% [markdown]
# # # Data Clean

# # %%
# # Load, drop unnecessary columns

# # Drop unnecessary columns
# columns_to_drop = ["comments", "Timestamp",'state']  # Add any other irrelevant columns
# df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

# # Check for null values
# print('Null values','\n',df_cleaned.isnull().sum())

# # # Fill Nulls
# # matches_cleaned['city']=matches_cleaned['city'].fillna("Unknown")
# # matches_cleaned['winner']=matches_cleaned['winner'].fillna("Draw")
# #Check Duplicates
# print("Duplicates in mental health:",df_cleaned.duplicated().sum())

# # %% [markdown]
# # # Selecting the features and target

# # %%
# # Select the features we want to use and the target variable

# features = ['Gender','wellness_program','benefits']
# target = 'treatment'

# # %% [markdown]
# # # Some Analysis

# # %%
# #Check Duplicates

# print("Duplicates in mental health:",df_cleaned.duplicated().sum())
# print(df_cleaned.head(),'\n')
# print(df_cleaned.tail())

# # %% [markdown]
# # # Separate the features x and the target y

# # %%
# # Separate the features (x) and the target(y)
# X = df[features].copy()
# y = df[target].copy()

# # Convert Yes/No to 1/0
# y = y.map(lambda val: 1 if str(val).strip().lower().startswith('y') else 0)

# # X = df_cleaned[features]
# # y = df_cleaned[target]
# print(X.shape)              # Should be (1259, 3)
# print(len(features))        # Should also be 3

# # Revisting the data Again
# print(df_cleaned.head(),'\n')
# print(df_cleaned.tail())

# # %% [markdown]
# # # encoding the data (categorical data)

# # %%
# encoder = OneHotEncoder(drop='first')  # drop='first' to avoid dummy variable trap
# X_encoded = encoder.fit_transform(X).toarray()

# # Convert back to DataFrame with proper column names
# encoded_cols = encoder.get_feature_names_out(features)
# X_encoded = pd.DataFrame(X_encoded, columns=encoded_cols)

# # %%
# print(X_encoded.shape)
# print(df_cleaned[features].isna().sum())

# # %% [markdown]
# # #  Split the data and train the model

# # %%
# # # Split the data into a training set (80%) and a testing set (20%)
# # # random_state ensures the split is the same every time we run the code
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)   
# # # Create an instance of the Logistic Regression model

# # # Train the model using the training data
# # lasso.fit(X_train, y_train)
# # lasso = Lasso(alpha=0.1)  # alpha controls regularization strength
# # lasso.fit(X_train, y_train)
# # Split the data into a training set (80%) and a testing set (20%)
# # random_state ensures the split is the same every time we run the code
# LR = LogisticRegression(
#     penalty='l1',
#     solver='saga',
#     C=1.0,          # inverse of regularization strength
#     max_iter=10000,
#     random_state=42
# )
# LR.fit(X_train, y_train)
# y_pred = LR.predict(X_test)

# # %% [markdown]
# # #  Predictions

# # %%
# # y_pred = lasso.predict(X_test)
# y_pred = LR.predict(X_test)  # ✅ gives 0/1 labels for classification metrics

# # %% [markdown]
# # #  Final Metrics

# # %%
# # mae = mean_absolute_error(y_test, y_pred)
# # mse = mean_squared_error(y_test, y_pred)
# # rmse = np.sqrt(mse)
# # r2 = r2_score(y_test, y_pred)

# # print(f"Mean Absolute Error (MAE): {mae:.4f}")
# # print(f"Mean Squared Error (MSE): {mse:.4f}")
# # print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# # print(f"R² Score: {r2:.4f}")
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
# print(f"Precision: {precision_score(y_test, y_pred):.4f}")
# print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
# print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# from sklearn.metrics import classification_report, confusion_matrix

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))



#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in th
#e read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
# for filename in filenames:
        
#     print(os.path.join(dirname, filename))

import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# file_path = os.path.join(os.path.dirname(_file_), "survey.csv")
# df = pd.read_csv(file_path)


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # BACK TO THE CLASSIFICATION TASK 

# In[2]:


import kagglehub

def run():
    # Download latest version
    # file_path = os.path.join(os.path.dirname(_file_), "survey.csv")
    # file_path = os.path.join(os.path.dirname(_file_), "survey", "survey.csv")

    df = pd.read_csv(r"C:\Users\acer\Downloads\capestone_main\survey.csv")

    # # LOAD THE KNIVES ...IMPORT LIBRARIES

    # In[3]:


    
    # # step 2.Load the DataSet

    # In[4]:


    # df = pd.read_csv('survey.csv')

    # 

    # In[5]:


    print("Mental Health Shape:",df.shape)

    # info
    df.info
    df.describe(include='all')
    # NULL VALUES
    print("Mental Health  nulls:\n",df.isnull().sum())

    # # Lets take a Quick Look

    # # Data Cleaning

    # In[6]:


    # Load, drop unnecessary columns

    # Drop unnecessary columns
    columns_to_drop = ["comments", "Timestamp",'state']  # Add any other irrelevant columns
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

    # Check for null values
    print('Null values','\n',df_cleaned.isnull().sum())

    # # Fill Nulls
    # matches_cleaned['city']=matches_cleaned['city'].fillna("Unknown")
    # matches_cleaned['winner']=matches_cleaned['winner'].fillna("Draw")
    #Check Duplicates
    print("Duplicates in mental health:",df_cleaned.duplicated().sum())

    # # Step 4. Selecting the features and target

    # In[7]:


    # Select the features we want to use and the target variable

    features = ['Gender','wellness_program','benefits']
    target = 'treatment'


    # # Step 5. Some Analysis

    # In[8]:



    #Check Duplicates

    print("Duplicates in mental health:",df_cleaned.duplicated().sum())
    print(df_cleaned.head(),'\n')
    print(df_cleaned.tail())

    # # Step 6. Separate the features (x) and the target(y)

    # In[9]:



    # Separate the features (x) and the target(y)
    X = df[features].copy()
    y = df[target].copy()

    # Convert Yes/No to 1/0
    y = y.map(lambda val: 1 if str(val).strip().lower().startswith('y') else 0)

    # X = df_cleaned[features]
    # y = df_cleaned[target]
    print(X.shape)              # Should be (1259, 3)
    print(len(features))        # Should also be 3

    # Revisting the data Again
    print(df_cleaned.head(),'\n')
    print(df_cleaned.tail())

    # #  Step 7 Encoding the data (CATEGORICAL DATA)

    # In[10]:


    encoder = OneHotEncoder(drop='first')  # drop='first' to avoid dummy variable trap
    X_encoded = encoder.fit_transform(X).toarray()

    # Convert back to DataFrame with proper column names
    encoded_cols = encoder.get_feature_names_out(features)
    X_encoded = pd.DataFrame(X_encoded, columns=encoded_cols)


    # In[11]:


    print(X_encoded.shape)
    print(df_cleaned[features].isna().sum())

    # # Step 8. Split the data and train the model

    # In[12]:


    # # Split the data into a training set (80%) and a testing set (20%)
    # # random_state ensures the split is the same every time we run the code
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)   
    # # Create an instance of the Logistic Regression model

    # # Train the model using the training data
    # lasso.fit(X_train, y_train)
    # lasso = Lasso(alpha=0.1)  # alpha controls regularization strength
    # lasso.fit(X_train, y_train)
    # Split the data into a training set (80%) and a testing set (20%)
    # random_state ensures the split is the same every time we run the code
    LR = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=1.0,          # inverse of regularization strength
        max_iter=10000,
        random_state=42
    )
    LR.fit(X_train, y_train)
    import pickle
    with open("classification.pkl", "wb") as f:
        pickle.dump(LR, f) 


                            
     # replace model with your classifier variable name
    print("✅ classification.pkl saved successfully")

    y_pred = LR.predict(X_test)



    # # Step 9. Predictions

    # In[13]:


    # y_pred = lasso.predict(X_test)
    y_pred = LR.predict(X_test)  # ✅ gives 0/1 labels for classification metrics


    # # Step 10. Final Metrics

    # In[14]:


    # mae = mean_absolute_error(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_test, y_pred)

    # print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # print(f"Mean Squared Error (MSE): {mse:.4f}")
    # print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    # print(f"R² Score: {r2:.4f}")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    from sklearn.metrics import classification_report, confusion_matrix

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
