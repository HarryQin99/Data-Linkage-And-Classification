import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from statistics import variance

# function used to determine the type of input 
# is float or not
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# read the world.csv and life.csv
world = pd.read_csv('world.csv',encoding= 'ISO-8859-1')
life = pd.read_csv('life.csv',encoding= 'ISO-8859-1')

# preprocessing the 'Time' column in the world
world_new = world.copy()
world_new.Time = pd.to_numeric(world_new['Time'],errors='coerce')
world_new['Time'] = world_new['Time'].fillna("0")
world_new['Time']=world_new['Time'].astype(int)

# creat a column called 'Time' in life, which has
# same value with 'Year' column for each row
life_new = life.copy()
life_new['Time'] = life.loc[:,'Year']
life_new['Time']=life_new['Time'].astype(int)

# merge two dataframe 'world_new' and 'life_new'
# based on 'Country Code' and 'Time'
merged = world_new.merge(life_new, on=['Country Code','Time'])

# extract all the useful datas for classification
data_no_using = ['Country','Year','Country Name','Time','Country Code','Life expectancy at birth (years)']
data_using = []
for data in merged:
    if data not in data_no_using:
        data_using.append(data)
datas = merged[data_using]
        
# get just the class labels
classlabel = merged['Life expectancy at birth (years)']

# randomly select 2/3 of the instances to be training and the rest to be testing
X_train, X_test, y_train, y_test = train_test_split(datas ,classlabel, train_size=2/3, test_size=1/3, random_state=100)

# figure out the median for each column 
# used for imputation
dictionary_data_info = {}
for data in X_train:
    value_list = []
    for value in X_train[data]:
        if isfloat(value):
            value_list.append(float(value))
    value_median = np.median(value_list)
    dictionary_data_info[data] = [value_median]
    
# replace all the '..' in X_train with the median of that column
X_train = X_train.applymap(lambda x: np.nan if not isfloat(x) else x)
for data in X_train:
    X_train[data].fillna(dictionary_data_info[data][0], inplace=True)
    X_train[data] = X_train[data].astype('float')
    
# replace all the '..' in X_train with the median of that column
X_test = X_test.applymap(lambda x: np.nan if not isfloat(x) else x)   
for data in X_test:
    X_test[data].fillna(dictionary_data_info[data][0], inplace=True)
    X_test[data] = X_test[data].astype('float')

# find the mean and variance, median for each
# columns
for data in X_train:
    value_list = []
    for value in X_train[data]:
        if isfloat(value):
            value_list.append(float(value))
    value_mean = sum(value_list)/len(value_list)
    value_median = np.median(value_list)
    value_variance = variance(value_list)
    dictionary_data_info[data] = [value_median,value_mean,value_variance]
    
# normalise the data to have 0 mean and unit variance
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train) 
X_test=scaler.transform(X_test)

# build the model by k-NN classiﬁer on the train set
# and evaluation the accuracy by the test set, the k here
# is 5
knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
y_pred=knn5.predict(X_test)
accurate_knn5 = accuracy_score(y_test, y_pred)

# build the model by k-NN classiﬁer on the train set
# and evaluation the accuracy by the test set, the k here
# is 5
knn10 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)
y_pred=knn10.predict(X_test)
accurate_knn10 = accuracy_score(y_test, y_pred)

# build the model by Decision tree classiﬁer on the train set
# and evaluation the accuracy by the test set, the maximum depth 
# here is 4
dt = DecisionTreeClassifier(random_state=100, max_depth=4)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
accurate_dt = accuracy_score(y_test, y_pred)

# store all median, mean, variance for each columns
# into big list
total_info = []
for key in dictionary_data_info.keys():
    info = []
    info.append(key)
    for value in dictionary_data_info[key]:
        info.append(round(value,3))
    total_info.append(info)

# store the list called total_info as a dataframe
# and convert to csv file called 'task2a.csv'
col_name = ['feature','median','mean','variance']
task2a = pd.DataFrame(total_info,columns = col_name)
task2a.to_csv('task2a.csv',index = False) 

# print out the accuracy of each classifier
print('Accuracy of decision tree:' +str(round(accurate_dt,3)))
print('Accuracy of k-nn (k=5):'+str(round(accurate_knn5,3)))
print('Accuracy of k-nn (k=10):'+str(round(accurate_knn10,3)))