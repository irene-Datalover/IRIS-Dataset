#!/usr/bin/env python
# coding: utf-8

# # Introduction

# 
# The Iris dataset is a classic dataset in the field of machine learning and statistics. It consists of 150 samples of iris flowers, each belonging to one of three species: Setosa, Versicolor, and Virginica. Each sample includes four features: sepal length, sepal width, petal length, and petal width, all measured in centimeters.
# 

# # Importing Necessary libraries

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading IRIS dataset 

# In[2]:


file_path=r"C:\Users\Irene Chelsia\Downloads\Iris.csv"
df=pd.read_csv(file_path)
df


# # Data Understanding

# In[3]:


df.head()


# In[4]:


df.tail()


# # 

# In[5]:


df.describe()


# In[6]:


#Statistical description of the data
df.describe().plot()


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


df.columns


# In[10]:


df.shape


# In[11]:


len(df)


# In[12]:


df['Species'].unique()
d2=df['Species'].value_counts()
plt.pie(d2,labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],autopct='%1.1f%%')
plt.title('Species Frequency')
plt.show()


# In[13]:


df['Species'].nunique()


# In[14]:


df.duplicated().sum()


# In[15]:


df[df.duplicated()]


# In[16]:


df.isnull().sum()


# In[17]:


df.drop("Id", axis=1).boxplot(by="Species", figsize=(10, 10))
plt.show()


# In[18]:


df.corr()


# In[19]:


iris=pd.read_csv(file_path)

ax = iris[iris.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                    color='red', label='setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                color='green', label='versicolor', ax=ax)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 
                                                color='blue', label='virginica', ax=ax)
ax.set_title("Sepal Length Vs Sepal Width")


# In[20]:


corr=df.corr()
sns.heatmap(corr,annot=True)


# # Data Preprocessing

# In[21]:


df.drop(columns=['Id'],inplace=True)
df.head()


# In[22]:


df[['Prefix','Species']]=df['Species'].str.split('-',1,expand=True)


# In[23]:


df.drop('Prefix',axis=1,inplace=True)


# In[24]:


df['Sepal ratio']=df['SepalLengthCm']/df['SepalWidthCm']
df['Petal ratio']=df['PetalLengthCm']/df['PetalWidthCm']


# In[25]:


df.head()


# # Feature Engineering

# In[26]:


minmax=MinMaxScaler()
df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Sepal ratio','Petal ratio']]=minmax.fit_transform(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Sepal ratio','Petal ratio']])

df.head()


# In[27]:


encoder=LabelEncoder()
df['Species']=encoder.fit_transform(df['Species'])

df.head()


# In[28]:


labels = ['Small', 'Medium', 'Tall']
df['Sepal bin'] = pd.cut(df['Sepal ratio'], bins=3, labels=labels)
df['Petal bin'] = pd.cut(df['Petal ratio'], bins=3, labels=labels)

df.head()


# # Split the data into training and testing sets

# In[29]:


X=df.iloc[:,:4]
y=df.iloc[:,4:5]


# In[30]:


X


# In[31]:


y


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# # Visualization

# In[33]:


sns.pairplot(df, hue='Species', markers=['o', 's', 'D'])


# In[34]:


plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='Species', y=feature, data=df, palette='Set3')
    plt.title(feature)
plt.tight_layout()  


# # Random Forest Classifier

# In[35]:


rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)


# In[36]:


y_pred=rfc.predict(X_test)


# # Model Performance

# In[37]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[38]:


precision = precision_score(y_test, y_pred, average='weighted')
print("Precision:", precision)


# In[39]:


recall = recall_score(y_test, y_pred, average='weighted')
print("Recall:", recall)


# In[40]:


f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)


# In[41]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[42]:


roc_auc = roc_auc_score(y_test, rfc.predict_proba(X_test), multi_class='ovr')
print("ROC AUC Score:", roc_auc)


# # Insights

"""Species Differentiation: The dataset helps distinguish between three types of iris flowers: Setosa, Versicolor, and Virginica, based on their petal and sepal measurements.

Feature Importance: It shows which characteristics (sepal length, sepal width, petal length, petal width) are most relevant for classifying iris species.

Model Selection: It aids in choosing the right machine learning model (like decision trees or support vector machines) for accurately predicting iris species.

Model Evaluation: It helps assess how well the chosen model performs in predicting iris species using metrics like accuracy or confusion matrix.

Real-World Applications: Insights from the dataset can be applied to various fields such as botany or agriculture for species identification and classification."""
