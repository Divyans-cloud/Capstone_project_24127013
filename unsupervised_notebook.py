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
# # # Back To Unsupervised Task ...

# # %% [markdown]
# # # so go with PCA task and Soon Jump into Clustering Master (k-Means)

# # %%
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.pipeline import Pipeline
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # %% [markdown]
# # # Step 2.Load the Data and Drop Unnecessary data

# # %%
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

# print("Path to dataset files:", path)

# # %%
# # Load your dataset
# df = pd.read_csv("/kaggle/input/mental-health-in-tech-survey/survey.csv")

# # Drop non-numeric or unnecessary columns
# df = df.drop(columns=["timestamp", "state", "comments"], errors="ignore")

# # %%
# print("\nOriginal Dataset:")
# # Quick look
# df.head() # first five rows
# print("Mental Health Shape:",df.shape)

# # info
# df.info
# df.describe(include='all')
# # NULL VALUES
# print("Mental Health  nulls:\n",df.isnull().sum())

# # %% [markdown]
# # # Step 3 Encode Categorical Column

# # %%



# # Selecting features
# # If df is your dataframe
# X = df.drop(columns=["timestamp", "state", "comments"], errors="ignore") # drop unnecessary columns
# # Drop columns that have datetime strings
# df = df.drop(columns=['Timestamp'])

# # OR, if you want to drop all columns with datetime dtype
# df = df.select_dtypes(exclude=['datetime64[ns]', 'object'])  
# X = df.copy()
# # Identify categorical columns
# categorical_cols = X.select_dtypes(include=['object']).columns

# # Column transformer for encoding
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(drop='first'), categorical_cols)
#     ],
#     remainder='passthrough'
# )

# X_encoded = preprocessor.fit_transform(X)

# # %% [markdown]
# # # Step 3.Standardize the data:
# #     PCA is affected by scale. We need to scale the features so each contributes equally.

# # %%
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(df)

# # %% [markdown]
# # # Step 4.Apply PCA and Encoding Dataset

# # %%
# # Detect categorical & numeric columns
# cat_cols = X.select_dtypes(include=['object']).columns
# num_cols = X.select_dtypes(exclude=['object']).columns

# # Preprocessing pipeline

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
#     ]
# )

# # PCA + KMeans Pipeline


# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("pca", PCA(n_components=1, random_state=42)),
# ])

# # Transform data
# X_pca = pipeline.fit_transform(X)

# # %% [markdown]
# # # Step 5.Actual Hero k-Means Clustering & ELBOW METHOD :
# #     
# #     clearly by observing Elbow Method Graph we can figure out optimal value of k as 2

# # %%
# # Elbow method to find optimal clusters

# inertia = []
# K_range = range(1, 11)
# for k in K_range:
#     km = KMeans(n_clusters=k, random_state=42)
#     km.fit(X_pca)
#     inertia.append(km.inertia_)

# plt.plot(K_range, inertia, marker="o")
# plt.xlabel("Number of clusters")
# plt.ylabel("Inertia")
# plt.title("Elbow Method")
# plt.show()

# # Fit KMeans with optimal clusters (example: 3,2,4...)

# optimal_k = 2  # Change based on elbow plot (tried with different optimal value  but according to elbow plot 2 is finalised)
# kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# labels = kmeans.fit_predict(X_pca)

# # Fit KMeans
# optimal_k = 2
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# labels = kmeans.fit_predict(X_pca)

# # Visualize clusters
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("K-Means Clusters (PCA Reduced Data)")
# plt.grid()

#!/usr/bin/env python
# coding: utf-8

# In[146]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
import pandas as pd

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# # Back To Unsupervised Task ...

# # Since We have more Features we first go with:
#      PCA(principal component analysis) and reduce the features into components and PCA will handle the dimensionality reduction and take care of the “which features matter most” part.

# #  so go with PCA task and Soon Jump into Clustering Master (k-Means) 
# 

# 
# # Step 1. Importing Required Libraries

# In[147]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# # Step 2.Load the Data and Drop Unnecessary data 

# # Loading ...⏳ 🔄  🔃 🕔

# In[148]:


import kagglehub

def run():
    # Download latest version
    # path = kagglehub.dataset_download("osmi/mental-health-in-tech-survey")

    # print("Path to dataset files:", path)

    # In[149]:


    # Load your dataset
    # # df = pd.read_csv("/kaggle/input/mental-health-in-tech-survey/survey.csv")
    # file_path = os.path.join(os.path.dirname(_file_), "survey.csv")
    # file_path = os.path.join(os.path.dirname(_file_), "survey", "survey.csv")
    df = pd.read_csv(r"C:\Users\acer\Downloads\capestone_main\survey.csv")

    # Drop non-numeric or unnecessary columns
    df = df.drop(columns=["timestamp", "state", "comments"], errors="ignore")

    # In[150]:


    print("\nOriginal Dataset:")
    # Quick look
    df.head() # first five rows
    print("Mental Health Shape:",df.shape)

    # info
    df.info
    df.describe(include='all')
    # NULL VALUES
    print("Mental Health  nulls:\n",df.isnull().sum())


    # # Step 3 Encode Categorical Column

    # In[151]:


    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer


    # Selecting features
    # If df is your dataframe
    X = df.drop(columns=["timestamp", "state", "comments"], errors="ignore") # drop unnecessary columns
    # Drop columns that have datetime strings
    df = df.drop(columns=['Timestamp'])

    # OR, if you want to drop all columns with datetime dtype
    df = df.select_dtypes(exclude=['datetime64[ns]', 'object'])  
    X = df.copy()
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Column transformer for encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_encoded = preprocessor.fit_transform(X)


    # # Step 3.Standardize the data

    #      :
    #      PCA is affected by scale. We need to scale the features so each contributes equally.

    # In[152]:


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # # Step 4.Apply PCA and Encoding Dataset

    # In[153]:


    # Detect categorical & numeric columns
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(exclude=['object']).columns

    # Preprocessing pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    # PCA + KMeans Pipeline


    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=2, random_state=42)),
    ])

    # Transform data
    X_pca = pipeline.fit_transform(X)

    # # Step 5.Actual Hero k-Means Clustering & ELBOW METHOD

    # #            :
    #      clearly by observing Elbow Method Graph we can figure out optimal value of k as 2

    # In[154]:



    # Elbow method to find optimal clusters

    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_pca)
        inertia.append(km.inertia_)

    plt.plot(K_range, inertia, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    # Fit KMeans with optimal clusters (example: 3,2,4...)

    optimal_k = 2  # Change based on elbow plot (tried with different optimal value  but according to elbow plot 2 is finalised)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    # Fit KMeans
    optimal_k = 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    import pickle
    with open("unsupervised.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    print("✅ unsupervised.pkl saved successfully")

    # Visualize clusters
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.7)
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.title("K-Means Clusters (PCA Reduced Data)")
    # plt.grid()
    plt.scatter(X_pca[:, 0], c=labels, cmap="viridis", alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clusters (PCA Reduced Data)")
    plt.grid()
    # # In[ ]:
