# Azure Bootcamp 2018 - ValueMomentum
## Azure Machine Learning Services

### @AakashSinha2018

## Complete Tutorial on Azure Machine Learning Services (PDF)

https://opdhsblobprod01.blob.core.windows.net/contents/
4a6d75bb3af747de838e6ccc97c5d978/ce7aa906e78f42e8a43cc95f71c4bdae?
sv=2015-04-05&sr=b&sig=FQZhStzcqKucXCjrsVibhNulQAoLq%2B5H%2BrF5kqP66xQ%3D&st=
2018-05-02T06%3A30%3A28Z&se=2018-05-03T06%3A40%3A28Z&sp=r

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

from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run

### Import important libraries required for building the model

from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import os
import pickle
