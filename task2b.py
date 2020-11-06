import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math,random
from scipy.spatial.distance import pdist, squareform
from statistics import variance

# function used to determine the type of input 
# is float or not
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

# function uesd for VAT visualisation
def VAT(R):
    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))
        
    J = list(range(0, N))
    
    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)


    I = i[j]
    del J[I]

    y = np.min(R[I,J], axis=0)
    j = np.argmin(R[I,J], axis=0)
    
    I = [I, J[j]]
    J = [e for e in J if e != J[j]]
    
    C = [1,1]
    for r in range(2, N-1):   
        y = np.min(R[I,:][:,J], axis=0)
        i = np.argmin(R[I,:][:,J], axis=0)
        j = np.argmin(y)        
        y = np.min(y)      
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])
    
    y = np.min(R[I,:][:,J], axis=0)
    i = np.argmin(R[I,:][:,J], axis=0)
    
    I.extend(J)
    C.extend(i)
    
    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx

    RV = R[I,:][:,I]
    
    return RV.tolist(), C, I

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
X_train, X_test, y_train, y_test = train_test_split(datas ,classlabel, train_size=2/3, test_size=1/3, random_state=42)

# calculate the median, mean, variance for the
# data of each columns and store into a dictonary
dictionary_data_info = {}
for data in X_train:
    value_list = []
    for value in X_train[data]:
        if isfloat(value):
            value_list.append(float(value))
    value_mean = sum(value_list)/len(value_list)
    value_median = np.median(value_list)
    value_variance = variance(value_list)
    dictionary_data_info[data] = [value_median,value_mean,value_variance]
    
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

# store all the original features names 
# into a list
feature_list_original = []
for data in X_train:
    feature_list_original.append(data)

# copy X_train and X_test after perprocessing 
# used for feature engineering and selection via PCA
# and feature selection of first 4 features
X_train_f4 = X_train.copy()
X_test_f4 = X_test.copy()
X_train_pca = X_train.copy()
X_test_pca = X_test.copy()

# feature engineering using interaction term pairs for
# both train set and test set
column_num = len(X_train.columns)
for i in range(column_num):
    for j in range(i,column_num):
        if i != j:
            X_train[data_using[i]+' '+data_using[j]] = (X_train.iloc[:,i].astype('float'))*(X_train.iloc[:,j].astype('float'))
for i in range(column_num):
    for j in range(i,column_num):
        if i != j:
            X_test[data_using[i]+' '+data_using[j]] = (X_test.iloc[:,i].astype('float'))*(X_test.iloc[:,j].astype('float'))
# print the first five rows of trains set and 
# test after generating the interation terms
print('The first five rows of train set after feature engineering of interaction terms:')
print(X_train.head())
print('The first five rows of test set after feature engineering of interaction terms:')
print(X_test.head())
print('\n')

        
# store all the features names into a list
feature_list = []
for data in X_train:
    feature_list.append(data)
    
# build the model by k-nn classifier, k here is 5
knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)

# normalise the data to have 0 mean and unit variance
scaler =  preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train) 
X_test=scaler.transform(X_test)
X_test = pd.DataFrame(X_test)
X_train = pd.DataFrame(X_train)
X_train.columns = feature_list
X_test.columns = feature_list

# Apply VAT Algorithm to the X_train dataset and visualise using heatmap
RV, C, I = VAT(X_train[feature_list_original])
x=sns.heatmap(RV,cmap='viridis',xticklabels=False,yticklabels=False)
x.set(xlabel='Objects', ylabel='Objects')
plt.savefig('task2bgraph1.png')

# feature engineering using clustering label, k-means clustering
# the k here is 3, decided by the heatmap created above
kmean= KMeans(n_clusters = 5,random_state =42)
kmean = kmean.fit(X_train[feature_list_original]) 
X_train['classlabel'] = kmean.labels_
X_test['classlabel'] = kmean.predict(X_test[feature_list_original])
# print the first five rows of the train set and test set after
# generating the cluster label 
print('The first five rows of train set after feature engineering of clustering(first 210 features have been standarlized):')
print(X_train.head())
print('The first five rows of test set after feature engineering of clustering(first 210 features have been standarlized):')
print(X_test.head())
print('\n')

# feature filtering by Mutual information
MI_list = mutual_info_classif(X_train,y_train,random_state = 42)
MI_frame = pd.DataFrame(MI_list)
MI_frame = MI_frame.transpose()
MI_frame.columns = X_train.columns
print('Estimated mutual information between each features and the target:')
print(MI_frame)

# figure out the 4 most well correlated features
MI_frame1 = MI_frame.iloc[:,np.argsort(MI_frame.loc[0])]
features = MI_frame1.columns.tolist()
features.reverse()
features = features[:4]

# print out 4 most wel correlated features calculated 
# by Mutual information
print('4 most well correlated features got:')
for feature in features:
    print(feature)
print('\n')

# extract those features from X_train and
# X_test 
X_test_Mi = X_test[features]
X_train_Mi = X_train[features]
# print the first five rows of the train set and test set
# after the feature selection
print('The first five rows of train set after feature selection:')
print(X_train_Mi.head())
print('The first five rows of test set after feature selection:')
print(X_test_Mi.head())
print('\n')

# evaluate the performance of the model 
# build by those 4 features and k-nn classifier, k is 5
knn5.fit(X_train_Mi,y_train)
y_pred = knn5.predict(X_test_Mi)
accurate_fe = accuracy_score(y_test, y_pred)

# normalise the X_train_pca and X_test_pca
# to have 0 mean and unit variance
scaler = preprocessing.StandardScaler().fit(X_train_pca)
X_train_pca=scaler.transform(X_train_pca) 
X_test_pca=scaler.transform(X_test_pca)

# making the X_train_pca and X_test_pca to 4 dimensional data
# using PCA
pca = PCA(n_components=4)
sklearn_train = pca.fit(X_train_pca)
X_train_pca = pd.DataFrame(sklearn_train.transform(X_train_pca),columns= ['1st principle component','2nd principle component','3rd principle component','4th principle component'])
X_test_pca = pd.DataFrame(sklearn_train.transform(X_test_pca),columns= ['1st principle component','2nd principle component','3rd principle component','4th principle component'])
# print the first five rows of train set and test set 
# after pca reducing the dimensions
print('The first five rows of train set after PCA:')
print(X_train_pca.head())
print('The first five rows of test set after PCA:')
print(X_test_pca.head())
print('\n')

# evaluate the performance of the model 
# build by pca(first 4 pcs) and k-nn classifier, k is 5
knn5.fit(X_train_pca,y_train)
y_pred = knn5.predict(X_test_pca)
accurate_pca = accuracy_score(y_test, y_pred)

# extract the first 4 features
data_4 = []
i = 0
for data in X_train_f4:
    data_4.append(data)
    i +=1
    if i == 4:
        break
X_train_f4 = X_train_f4[data_4]
X_test_f4 = X_test_f4[data_4]

# normalise the X_train_pca and X_test_pca
# to have 0 mean and unit variance
scaler = preprocessing.StandardScaler().fit(X_train_f4)
X_train_f4 =scaler.transform(X_train_f4) 
X_test_f4 =scaler.transform(X_test_f4)
X_train_f4 = pd.DataFrame(X_train_f4)
X_test_f4 = pd.DataFrame(X_test_f4)
X_train_f4.columns = data_4
X_test_f4.columns = data_4
# print the train set and test set for choosing
# first 4 features to build and test the model
print('The first five rows of train set after naive feature selection:')
print(X_train_f4.head())
print('The first five rows of test set after naive feature selection:')
print(X_test_f4.head())
print('\n')

# evaluate the performance of the model 
# build by first 4 features and k-nn classifier, k is 5
knn5.fit(X_train_f4,y_train)
y_pred = knn5.predict(X_test_f4)
accurate_f4 = accuracy_score(y_test, y_pred)

# print out the accuracy for thoes three method
print('Accuracy of feature engineering: '+str(round(accurate_fe,3)))
print('Accuracy of PCA: '+str(round(accurate_pca,3)))
print('Accuracy of first four features: '+str(round(accurate_f4,3)))

# the following code is for the suggestion part
# of the report
MI_list = mutual_info_classif(X_train,y_train,random_state = 42)
MI_frame = pd.DataFrame(MI_list)
MI_frame = MI_frame.transpose()
MI_frame.columns = X_train.columns
MI_frame1 = MI_frame.iloc[:,np.argsort(MI_frame.loc[0])]
features_or = MI_frame1.columns.tolist()
features_or.reverse()
accuracy_list = []
n_value = []
# calculate the accuracy of the model
for n in range(1,len(features_or)):
    n_value.append(n)
    features = features_or[:n]
    X_test_Mi = X_test[features]
    X_train_Mi = X_train[features]
    knn5.fit(X_train_Mi,y_train)
    y_pred = knn5.predict(X_test_Mi)
    accuracy_list.append(accuracy_score(y_test, y_pred))
plt.figure(2)  
plt.plot(n_value,accuracy_list)
plt.xlabel('Most n well correlated features')
plt.ylabel('Accuracy')
plt.savefig('task2bgraph2.png')
