# Azure Bootcamp 2018 - ValueMomentum
## Azure Machine Learning Services

#### @AakashSinha2018

## Complete Tutorial on Azure Machine Learning Services (PDF)

https://opdhsblobprod01.blob.core.windows.net/contents/4a6d75bb3af747de838e6ccc97c5d978/ce7aa906e78f42e8a43cc95f71c4bdae?sv=2015-04-05&sr=b&sig=FQZhStzcqKucXCjrsVibhNulQAoLq%2B5H%2BrF5kqP66xQ%3D&st=2018-05-02T06%3A30%3A28Z&se=2018-05-03T06%3A40%3A28Z&sp=r

_____________________________________________________________________________________________

## Step By Step Tutorial 


### Tutorial 1 : Classify Iris - Preparing the data

https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-1

### Tutorial 2 : Classify Iris - Build a model

https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-2

### Tutorial 3 : Classify Iris: Deploy a model

https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-3

_____________________________________________________________________________________________

## About the Code (demo1.py) - Titanic Problem

#### Get the dataset from this link -> https://www.kaggle.com/c/titanic/data

#### Use the Azure Machine Learning data preparation package

from azureml.dataprep import package


#### Use the Azure Machine Learning data collector to log various metrics

from azureml.logging import get_azureml_logger <br/>
from azureml.dataprep.package import run <br/>

#### Import important libraries required for building the model

from sklearn.preprocessing import Imputer <br/>
import pandas as pd <br/>
import numpy as np <br/>
from sklearn.linear_model import LogisticRegression <br/>
import sys <br/>
from sklearn import preprocessing <br/>
from sklearn.model_selection import train_test_split <br/>
from sklearn.metrics import classification_report <br/>
from sklearn.metrics import confusion_matrix <br/>
from sklearn.metrics import precision_recall_curve <br/>
import os <br/>
import pickle <br/>

#### Initializing the Logger (Azure ML Logger will keep track of all the activities/modifications taking place during the session.
run_logger = get_azureml_logger()

#### Creating an output folder for storing the output files (eg - pickle file, json file) 
os.makedirs('E:/Azure Boot Camp/Project/Demo Project/Demo1/Demo1/outputs', exist_ok=True)

#### This call will load the referenced package and return a DataFrame.If run in a PySpark environment, this call returns a Spark DataFrame. If not, it will return a Pandas DataFrame.

#### Loading the Titanic Dataset 
titanic_train = run('demo1.dprep', dataflow_idx=0, spark=False)

#### Obtaining the shape of data (Total Number of Rows and Columns)
print ('Titanic dataset shape: {}'.format(titanic_train.shape))

#### Exlporing Missing Data
titanic_train.apply(lambda x : sum(x.isnull()))

### Embarked (Boarding Point) - Replacing all the special characters, if any, with NaN values and then replacing NaN value with 'S' (Maximum Occurance)

titanic_train['Embarked'].replace('r^\s+$',np.NaN,regex=True,inplace=True) <br/>
titanic_train['Embarked'].replace(np.NaN,'S',regex=True,inplace=True) <br/>


#### Age (Age of the Passenger) - Imputer is used when we are not aware of which technique to use for replacing missing values. By using imputer -  If “mean”, then replace missing values using the mean along the axis. If “median”, then replace missing values using the median along the axis. If “most_frequent”, then replace missing using the most frequent value along the axis.

imp = Imputer(missing_values="NaN",strategy='mean',axis=0) <br/>
imp.fit(titanic_train[['Age']]) <br/>
titanic_train['Age'] = imp.fit_transform(titanic_train[['Age']]).ravel() <br/>

#### Creating a new feature - Family Size (Single, Small or Large) depending upon the number of members in family.

def size_to_type(x): <br/>
<br/>
    if(x == 1):  <br/>
<br/>
        return 'Single' <br/>
<br/>
    elif(x >= 2 and x <= 4):  <br/>
<br/>
        return 'Small' <br/>
<br/>
    else:  <br/>
<br/>
        return 'Large' <br/>

#### FamilySize = SibSp (Sibling, Spouse) + Parch (Parent, Child) + 1 (Self)

titanic_train['FamilySize'] = titanic_train.SibSp + titanic_train.Parch + 1 <br/>

titanic_train['FamilyType'] = titanic_train['FamilySize'].map(size_to_type) <br/>

#### Processing names of Passengers. Creating a dictionary for mapping similar Titles to a more generalized Title.

title_Dictionary = { <br/>

                        "Capt":       "Officer", "Col":        "Officer", <br/>

                        "Major":      "Officer", "Jonkheer":   "Royalty", <br/>

                        "Don":        "Royalty", "Sir" :       "Royalty", <br/>

                        "Dr":         "Officer", "Rev":        "Officer", <br/>

                        "the Countess":"Royalty","Dona":       "Royalty", <br/>
  
                        "Mme":        "Mrs", "Mlle":       "Miss", <br/>

                        "Ms":         "Mrs", "Mr" :        "Mr", <br/>

                        "Mrs" :       "Mrs", "Miss" :      "Miss", <br/>

                        "Master" :    "Master", "Lady" :      "Royalty" <br/>

} <br/>

#### Mapping the titles with the Dictionary Created

def extract_title(name): <br/>

    return name.split(',')[1].split('.')[0].strip() <br/>

titanic_train['Title'] = titanic_train['Name'].map(extract_title) <br/>

titanic_train['Title'] = titanic_train['Title'].map(title_Dictionary) <br/>

#### Ticket is a Alpha-Numeric Column, so stripping off the characters and only retrieving the numbers for each Ticket

def extract_number(ticket): <br/>
    if ticket.split(' '): <br/>
       return ticket.split(' ')[-1].strip() <br/>
    else : <br/>
     return ticket <br/>
      
titanic_train['Ticket_num'] = titanic_train['Ticket'].map(extract_number) <br/>

#### When there are a lot of missing values and cannot be subtituted with an appropriate value, we create dummies , i.e Transposing each category into a new column with 0/1 values. And we delete the column having missing values.

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'FamilyType', 'Embarked', 'Sex','Title']) <br/>

titanic_train1 = titanic_train1[titanic_train1.Ticket_num != 'LINE'] <br/>
titanic_train1.Ticket_num = titanic_train1.Ticket_num.astype(np.int64) <br/>

### Model Building

#### Target Variable is 'Survived' 

Y = titanic_train1['Survived'] <br/>

#### Dropping Target Variable, Unique Identifiers and Those attributes which are having categorical data

titanic_train1.drop(['PassengerId','Name','Cabin','Survived','Ticket'], axis=1, inplace=True) <br/>

#### Rest all data

X = titanic_train1 <br/>

#### Splitting Data into 70% (Train Data) and 30% (Test Data)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0) <br/>

print(X_train.info()) <br/>

#### Change regularization rate to get a different accuracy.

reg = 0.01 <br/>

#### Load regularization rate from argument if present

if len(sys.argv) > 1: <br/>
    reg = float(sys.argv[1]) <br/>

print("Regularization rate is {}".format(reg)) <br/>

#### Log the regularization rate

run_logger.log("Regularization Rate", reg) <br/>

#### Building Logistic Regression Model and fitting it

clf1 = LogisticRegression(C = 1/reg, random_state=0) <br/>
clf1.fit(X_train,Y_train) <br/>
print (clf1) <br/>

#### Evaluating the test set

accuracy = clf1.score(X_test, Y_test) <br/>
run_logger.log("Accuracy", accuracy) <br/>

#### Predictions

y_pred1 = clf1.predict(X_test) <br/>

#### Finding Confusion Matrix (Precision and Recall - A Performance Metric)

print(confusion_matrix(Y_test,y_pred1)) <br/>

#### Logging various performance metrics

y_scores = clf1.predict_proba(X_test) <br/>
precision, recall, thresholds = precision_recall_curve(Y_test, y_scores[:,1]) <br/>
run_logger.log("Precision", precision) <br/>
run_logger.log("Recall", recall) <br/>
run_logger.log("Thresholds", thresholds) <br/>

#### Serialize the model on disk in the special 'outputs' folder
#### The model is stored in a pickle file which can be used for further purposes without performing all the steps again

print ("Export the model to model.pkl") <br/>
f = open('E:/Azure Boot Camp/Project/Demo Project/Demo1/Demo1/outputs/model.pkl', 'wb') <br/>
pickle.dump(clf1, f) <br/>
f.close() <br/>

_____________________________________________________________________________________________

### For any doubts, you can reach out to - 
### 1. Harshal Wavre (harshal.wavre@valuemomentum.biz)
### 2. Aakash Sinha (aakash.sinha@valuemomentum.biz) or connect with me on LinkedIn - https://www.linkedin.com/in/aakashsinha19/


