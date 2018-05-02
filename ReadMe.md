# Azure Bootcamp 2018 - ValueMomentum
## Azure Machine Learning Services

#### @AakashSinha2018

## Complete Tutorial on Azure Machine Learning Services (PDF)

https://opdhsblobprod01.blob.core.windows.net/contents/4a6d75bb3af747de838e6ccc97c5d978/ce7aa906e78f42e8a43cc95f71c4bdae?sv=2015-04-05&sr=b&sig=FQZhStzcqKucXCjrsVibhNulQAoLq%2B5H%2BrF5kqP66xQ%3D&st=2018-05-02T06%3A30%3A28Z&se=2018-05-03T06%3A40%3A28Z&sp=r

______________________________________________________________________________________________________________

## Step By Step Tutorial 


### Tutorial 1 : Classify Iris - Preparing the data

https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-1

### Tutorial 2 : Classify Iris - Build a model

https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-2

### Tutorial 3 : Classify Iris: Deploy a model

https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-3

______________________________________________________________________________________________________________

## About the Code (demo1.py)

### Use the Azure Machine Learning data preparation package

from azureml.dataprep import package


### Use the Azure Machine Learning data collector to log various metrics

from azureml.logging import get_azureml_logger <br/>
from azureml.dataprep.package import run <br/>

### Import important libraries required for building the model

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

### Initializing the Logger (Azure ML Logger will keep track of all the activities/modifications taking place during the session.
run_logger = get_azureml_logger()

### Creating an output folder for storing the output files (eg - pickle file, json file) 
os.makedirs('E:/Azure Boot Camp/Project/Demo Project/Demo1/Demo1/outputs', exist_ok=True)

### This call will load the referenced package and return a DataFrame.If run in a PySpark environment, this call returns a Spark DataFrame. If not, it will return a Pandas DataFrame.

### Loading the Titanic Dataset 
titanic_train = run('demo1.dprep', dataflow_idx=0, spark=False)

### Obtaining the shape of data (Total Number of Rows and Columns)
print ('Titanic dataset shape: {}'.format(titanic_train.shape))

### Exlporing Missing Data
titanic_train.apply(lambda x : sum(x.isnull()))

### Embarked (Boarding Point) - Replacing all the special characters, if any, with NaN values and then replacing NaN value with 'S' (Maximum Occurance)

titanic_train['Embarked'].replace('r^\s+$',np.NaN,regex=True,inplace=True) <br/>
titanic_train['Embarked'].replace(np.NaN,'S',regex=True,inplace=True) <br/>


# Age (Age of the Passenger) - Imputer is used when we are not aware of which technique to use for replacing missing values. By using imputer -  If “mean”, then replace missing values using the mean along the axis. If “median”, then replace missing values using the median along the axis. If “most_frequent”, then replace missing using the most frequent value along the axis.

imp = Imputer(missing_values="NaN",strategy='mean',axis=0) <br/>
imp.fit(titanic_train[['Age']]) <br/>
titanic_train['Age'] = imp.fit_transform(titanic_train[['Age']]).ravel() <br/>





### For any doubts, reach out to me at aakash.sinha@valuemomentum.biz or connect with me on LinkedIn - https://www.linkedin.com/in/aakashsinha19/


