
import os
from github import Github, GithubException

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

df = pd.read_csv('VisitWithUs_project/data/tourism.csv')

#Data exploration
print(df.head())
print(df.info())
print(df.describe())
