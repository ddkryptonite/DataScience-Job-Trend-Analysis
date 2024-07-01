#supress warnings
import warnings
warnings.filterwarnings('ignore')

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Setting the Display to display all rows to prevent truncation because of the Large size of data 
pd.set_option('display.max_rows',None)

#Loading the data
DS=pd.read_csv('C:/Users/Administrator/Documents/Python Projects/Machine Learning/Data_Science_Jobs.csv')

#Inspecting Data
DS.head()

DS.info()

#Data Cleaning
#1.Rename Keywords to Min_Exp and Title2 to Company
DS.rename(columns={'Keywords':'Min_Exp', 'Title2':'Company'},inplace=True)

#Dropping Columns that are irrelevant and contain a large number of Null entries
DS.drop(['Salary','Type','Type13','starrating','View','Keywords5','Industry_Type_2','Employment_Type'], axis=1, inplace=True)

#3. Extracting only the minimum required years of experience
DS.Min_Exp=DS.Min_Exp.str.split('-').str[0]

#4.Cleaning the location Column
DS.Location=DS.Location.str.split('(').str[0]
DS.Location=DS.Location.str.split(',').str[0]
DS.Location=DS.Location.str.split('/').str[0]

DS.Location.value_counts().sort_index()

#Identifying similar cities with different names and renaming them
#The replace function is preferred to the rename function because the rename fuction is used to rename columns and not entries withi a column
DS.Location.replace({'Bangalore':'Bengaluru','Gurgaon Gurugram': 'Gurgaon','Delhi NCR':'Delhi'}, inplace=True)

#Assigning only non null values of Role Column to Dataframe
DS=DS[~pd.isnull(DS.Role)]
#Calculating the percentage of missing entries,if above line worked then Roleshouldbe 0.00 
round((DS.isnull().mean()*100),2)

##Assigning only non null values of Min_Exp Column to Dataframe
DS=DS[~pd.isnull(DS.Min_Exp)]
round(DS.isnull().mean()*100,2)

#Skills
Skills = pd.concat([DS[col] for col in DS.columns if col.startswith('Skill')], ignore_index=True)
Skills=Skills.str.lower()#converting all to lowercase
#Skills.value_counts()

Locs=pd.DataFrame(data=DS.Location.value_counts())
Locs.reset_index(inplace=True)
Locs.columns=['Location','counts']

import plotly.express as px

fig = px.pie(Locs, values='counts', names='Location',width=1000, height=600)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

DS.Min_Exp=DS.Min_Exp.astype(int)

plt.figure(figsize=(10,5))
ax=sns.countplot(x=DS.Min_Exp, palette='Set1')
total = len(DS)

for p in ax.patches:
    percentage = f'{100*p.get_height()/total:.1f}%\n'
    x = p.get_x() + p.get_width()/2
    y = p.get_height()
    ax.annotate(percentage, (x,y), ha='center', va='center')

plt.tight_layout
plt.show()

Locs=pd.DataFrame(data=DS.Role_Category.value_counts())
Locs.reset_index(inplace=True)
Locs.columns=['Role_Category','counts']

fig = px.pie(Locs, values='counts', names='Role_Category', height=1000, width=800)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

Skills = pd.concat([DS[col] for col in DS.columns if col.startswith('Skill')], ignore_index=True)
Skills=Skills.str.lower()

Locs = pd.DataFrame(data=Skills.value_counts().head(20))
Locs.reset_index(inplace=True)
Locs.columns=['Skillset','counts']

fig = px.pie(Locs, values='counts',names='Skillset',height=1000, width=800)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

sns.countplot(y=DS.Company, order=DS.Company.value_counts().iloc[:10].index)
plt.show()

fig = px.sunburst(DS, path=['Role_Category','Role'])
fig.update_layout(width=500,height=500)
fig.show()

fig = px.sunburst(DS, path=['Industry_Type_1','Role'])
fig.update_layout(width=600,height=600)
fig.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Encode categorical variables
encoded_df = pd.get_dummies(DS[['Role_Category', 'Functional_Area', 'Industry_Type_1', 'Title', 'Company']], drop_first=True)

# Define numerical features
numerical_features = DS[['Min_Exp']]

# Convert numerical features to a numpy array and ensure it is 2D
numerical_features_array = numerical_features.values.reshape(-1, 1)

# Combine encoded features with numerical features
X = np.hstack((encoded_df.values, numerical_features_array))

# Encode the target variable (Location)
y = DS['Location']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
