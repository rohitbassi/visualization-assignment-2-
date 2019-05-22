from sklearn import manifold

import pandas as pd
from flask import Flask
from flask import render_template
import random
import numpy.matlib as npm
import json
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from scipy import linalg as LA
import numpy as np
import sys
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, Response, jsonify
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
#making these as global variables that i can use in functions
df1 ="datafile.csv"
dfor=pd.read_csv(df1)#original file
df = pd.read_csv(df1)#another copy of file
random_samples=[]
kvalue=0
# print(len(df))
df = df.fillna(0)# filling all NAN to zero
size=500
#these 8 attributes i am working on
ftrs = ['b_impartial_courts', 'quartile', 'rank', 'a_government_consumption', 'b_transfers', 'c_gov_enterprises', '_size_government','g_restrictions_sale_real_property']
X=df[ftrs]
squaredLoadings=[]
pca_filter_attr=[]
list3=[]
#StandardScaler is that it will transform
# your data such that its distribution will have a mean value 0 and standard deviation of 1
scaler = StandardScaler()
df[ftrs] = scaler.fit_transform(df[ftrs])
features = df[ftrs]
data = np.array(features)

@app.route("/")
#this will render index.html with the start of server
def index():
    return render_template("index.html")
#RANDOM SAMPLING
def random_sampling():
    rand = random.sample(range(len(df)), size)
    for i in rand:
        random_samples.append(data[i])
#Kmeans and then getting elbow point and optimal K
def kelbow():
    #using kmeas method, inbuilt function in python
    distortions = []
    X=df[ftrs]
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append((k,sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]))
    coordinates=np.asarray(distortions)
    # print(coordinates)
    #this is method to find optimal K, refered this from the link professor gave
    first=coordinates[0]
    last=coordinates[-1]
    lineVec=last-first
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = coordinates - first
    scalarProduct = np.sum(vecFromFirst * npm.repmat(lineVecNorm, 9, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    # print(distToLine)
    # print(idxOfBestPoint)
    # plt.plot(K, distortions,'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()
    kvalue= distortions[idxOfBestPoint][0]
    print(kvalue)

    #dividing the data in K groups/clusters
    check=KMeans(n_clusters=kvalue).fit(X)
    list2=check.labels_
    # print(list2)
    df['kcluster'] = pd.Series(list2)
    clustering={}
    for i in range(len(list2)):
        try:
            clustering[list2[i]].append(i)   
        except KeyError:
            clustering[list2[i]]=[i]
    # print(clustering)
    # print (len(clustering[0]))
    # print (len(clustering[1]))
    # print (len(clustering[2]))
    #k=3
    global adaptive_samples
    #getting the cluster and storing the data in 3 clusters
    kcluster0 = df[df['kcluster'] == 0]
    kcluster1 = df[df['kcluster'] == 1]
    kcluster2 = df[df['kcluster'] == 2]
    #sampling the data with 40% samples
    temp1=kcluster0[ftrs].sample(frac=0.4)
    temp2=kcluster1[ftrs].sample(frac=0.4)
    temp3=kcluster2[ftrs].sample(frac=0.4)



    #SO this is my adaptive samples data!! yipee! -TASK1 done
    adaptive_samples = pd.concat([temp1, temp2, temp3])
    # print(len(adaptive_samples))

#This is calculation of eigen values and eigen vector using in built libraries
def vector(data):
    # print(data)
    #note-either correaltion or covariance will work,will get same result
    R = np.cov(data.T) # first find the covariance!
    print("-----")
    # print(R)
    evals, evecs = LA.eigh(R) #then calculate the eigenvalues and vectors
   # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    return evals,evecs

@app.route("/square")
def callsquare():
    # print("****")
    g,i,j,k,l=generate(data,3)
    valueandname=(j,k,l)
    print(valueandname)
    return pd.json.dumps(valueandname)
    # return pd.json.dumps(l)

@app.route("/squaresample")
def callsquare1():
    # print("****")
    g,i,j,k,l=generate1(adaptive_samples[ftrs],3)
    valueandname=(j,k,l)
    print(valueandname)
    return pd.json.dumps(valueandname)
    # return pd.json.dumps(l)

def generate1(data,k):
    [eigenValues, eigenVectors] = vector(data)
    # print(eigenVectors[0])
    squaredLoadings = []
    list2=[]
    counter = len(eigenVectors)
    for cols in range(0, counter):
        loadings = 0
        for row in range(0, k):
            loadings = loadings + eigenVectors[row][cols] * eigenVectors[row][cols]
        squaredLoadings.append(loadings)
    print(squaredLoadings)
    j1=max(squaredLoadings)
    i1=squaredLoadings.index(j1)
    list2.append(j1)
    squaredLoadings.remove(j1)
    j2=max(squaredLoadings)
    i2=squaredLoadings.index(j2)
    list2.append(j2)
    squaredLoadings.remove(j2)
    j3=max(squaredLoadings)
    i3=squaredLoadings.index(j3)
    list2.append(j3)
    squaredLoadings.remove(j3)
    j4=max(squaredLoadings)
    i4=squaredLoadings.index(j4)
    list2.append(j4)
    squaredLoadings.remove(j4)
    j5=max(squaredLoadings)
    i5=squaredLoadings.index(j5)
    list2.append(j5)
    squaredLoadings.remove(j5)
    j6=max(squaredLoadings)
    i6=squaredLoadings.index(j6)
    list2.append(j6)
    squaredLoadings.remove(j6)
    j7=max(squaredLoadings)
    i7=squaredLoadings.index(j7)
    list2.append(j7)
    squaredLoadings.remove(j7)
    j8=max(squaredLoadings)
    i8=squaredLoadings.index(j8)
    list2.append(j8)
    squaredLoadings.remove(j8)
    print(list2[:8])
    
    print(squaredLoadings,j1,i1,j2,i2,j3,i3)
    print("---")
    a=ftrs[i1]
    b=ftrs[i2]
    c=ftrs[i3]
    # print(ftrs[i1])
    return squaredLoadings,j1,list(ftrs),list(pca_filter_attr),list2

def generate11(data,k):
    [eigenValues, eigenVectors] = vector(data)
    # print(eigenVectors[0])
    squaredLoadings = []
    list2=[]
    counter = len(eigenVectors)
    for cols in range(0, counter):
        loadings = 0
        for row in range(0, k):
            loadings = loadings + eigenVectors[row][cols] * eigenVectors[row][cols]
        squaredLoadings.append(loadings)
    # print (eigenValues)
    # plt.plot(squaredLoadings)
    # plt.show()
    # print(squaredLoadings)
    # list3=[]
    # dataset=squaredLoadings
    # # list3=squaredLoadings
    # print(squaredLoadings)
    # j1=max(squaredLoadings)
    # i1=squaredLoadings.index(j1)
    # list2.append(j1)
    # list3.append(i1)
    # squaredLoadings.remove(j1)
    # j2=max(squaredLoadings)
    # i2=squaredLoadings.index(j2)
    # list2.append(j2)
    # list3.append(i2)
    # squaredLoadings.remove(j2)
    # j3=max(squaredLoadings)
    # i3=squaredLoadings.index(j3)
    # list2.append(j3)
    # list3.append(i3)
    # squaredLoadings.remove(j3)
    # j4=max(squaredLoadings)
    # i4=squaredLoadings.index(j4)
    # list2.append(j4)
    # list3.append(i4)
    # squaredLoadings.remove(j4)
    # j5=max(squaredLoadings)
    # i5=squaredLoadings.index(j5)
    # list2.append(j5)
    # list3.append(i5)
    # squaredLoadings.remove(j5)
    # j6=max(squaredLoadings)
    # i6=squaredLoadings.index(j6)
    # list2.append(j6)
    # list3.append(i6)
    # squaredLoadings.remove(j6)
    # j7=max(squaredLoadings)
    # i7=squaredLoadings.index(j7)
    # list2.append(j7)
    # list3.append(i7)
    # squaredLoadings.remove(j7)
    # j8=max(squaredLoadings)
    # i8=squaredLoadings.index(j8)
    # list2.append(j8)
    # list3.append(i8)
    # squaredLoadings.remove(j8)
    # print(list3)
    
    return squaredLoadings

#list1 gives the max of pca attributes
def generate(data,k):
    [eigenValues, eigenVectors] = vector(data)
    # print(eigenVectors[0])
    squaredLoadings = []
    list1=[]
    counter = len(eigenVectors)
    for cols in range(0, counter):
        loadings = 0
        for row in range(0, k):
            loadings = loadings + eigenVectors[row][cols] * eigenVectors[row][cols]
        squaredLoadings.append(loadings)
    # print (eigenValues)
    # plt.plot(squaredLoadings)
    # plt.show()
    # list3=[]
    print(squaredLoadings)
    # print(list3)
    j1=max(squaredLoadings)
    i1=squaredLoadings.index(j1)
    list1.append(j1)
    squaredLoadings.remove(j1)
    j2=max(squaredLoadings)
    i2=squaredLoadings.index(j2)
    list1.append(j2)
    squaredLoadings.remove(j2)
    j3=max(squaredLoadings)
    i3=squaredLoadings.index(j3)
    list1.append(j3)
    squaredLoadings.remove(j3)
    j4=max(squaredLoadings)
    i4=squaredLoadings.index(j4)
    list1.append(j4)
    squaredLoadings.remove(j4)
    j5=max(squaredLoadings)
    i5=squaredLoadings.index(j5)
    list1.append(j5)
    squaredLoadings.remove(j5)
    j6=max(squaredLoadings)
    i6=squaredLoadings.index(j6)
    list1.append(j6)
    squaredLoadings.remove(j6)
    j7=max(squaredLoadings)
    i7=squaredLoadings.index(j7)
    list1.append(j7)
    squaredLoadings.remove(j7)
    j8=max(squaredLoadings)
    i8=squaredLoadings.index(j8)
    list1.append(j8)
    squaredLoadings.remove(j8)
    print(list1[:8])
    # print(squaredLoadings,j1,i1,j2,i2,j3,i3)
    print("---")
    a=ftrs[i1]
    b=ftrs[i2]
    c=ftrs[i3]
    print("i amhere")
    print(list(ftrs))
    return squaredLoadings,j1,list(ftrs),list(pca_filter_attr),list1

#this function is used to calculate intrinstic dimensionality using scree plot of adaptive samples
@app.route("/screeR")
def scree_adaptive():

    [eigenValues, eigenVectors] = vector(adaptive_samples[ftrs])#ftrs is the attributes that is used in this assignment
    # chart_data = pd.json.dumps(eigenValues)
    # data = {'chart_data': chart_data}
    return pd.json.dumps(eigenValues)

#this function is used to calculate intrinsic dimensionality using screeplot of random samples
@app.route("/screeA")
def scree_adaptive1():
    [eigenValues, eigenVectors] = vector(df[ftrs])

    # chart_data = pd.json.dumps(eigenValues)
    # data = {'chart_data': chart_data}
    return pd.json.dumps(eigenValues)

#this function is used to run the task for top 3 pca attributes-plot scatter matrix (random case)
@app.route("/scattermatrixR")
def scatter_matrix_random():

    col_content = pd.DataFrame()
    for i in range(0, 3):
        col_content[ftrs[pca_filter_attr[i]]] = df[ftrs[pca_filter_attr[i]]][:size]
    
    col_content['clusterid'] = df['kcluster'][:size]
    print(pd.json.dumps(col_content))
    return pd.json.dumps(col_content)

#this function is used to run the task for top 3 pca attributes-plot scatter matrix (Adaptive case)
@app.route("/scattermatrixA")
def scatter_matrix_randomadap():
    
    col_content = pd.DataFrame()
    for i in range(0, 3):
        col_content[ftrs[pca_filter_attr[i]]] = adaptive_samples[ftrs[pca_filter_attr[i]]][:size]
    col_content['clusterid'] = df['kcluster'][:size]

    print(pd.json.dumps(col_content))
    return pd.json.dumps(col_content)


#this function is used for->  project into the top two PCA vectors via 2D scatterplot (random smaple)
@app.route('/scatterrandom')
def pcaR():
    col_content = []
    #basic implementation 

    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents
    #      , columns = ['principal component 1', 'principal component 2'])
    pca = PCA(n_components=2) #take n component as 2
    X = random_samples
    X = pca.fit_transform(X) 
    col_content = pd.DataFrame(X) #making coloums
    for i in range(0, 2): #loop upto 2
        #i have calculated the pca_filter_attr below in the code!
        #it contains the index of pca attri (with highest values indexes at the first)
        col_content[ftrs[pca_filter_attr[i]]] = dfor[ftrs[pca_filter_attr[i]]][:size]
    
    # return col_content.to_json
    return pd.json.dumps(col_content)

#this function is used for->  project into the top two PCA vectors via 2D scatterplot (adaptive sample)
@app.route('/randomsample1')
def pcaA():
    col_content = []
    #basic implementation 

    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(x)
    # principalDf = pd.DataFrame(data = principalComponents
    #      , columns = ['principal component 1', 'principal component 2'])
    X = adaptive_samples[ftrs]
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    col_content = pd.DataFrame(X)
    for i in range(0, 2):
        #i have calculated the pca_filter_attr below in the code!
        #it contains the index of pca attri (with highest values indexes at the first)
        col_content[ftrs[pca_filter_attr[i]]] = dfor[ftrs[pca_filter_attr[i]]][:size]
    return pd.json.dumps(col_content)


#this function is used to implement MDS correlation (Random samples)
@app.route('/mdscorrR')
def mdscorrR():
    col_content = []
        #Dissimilarity measure to use:
# ‘euclidean’:Pairwise Euclidean distances between points in the dataset.
# ‘precomputed’:Pre-computed dissimilarities are passed directly to fit and fit_transform
    dataset1 = manifold.MDS(n_components=2, dissimilarity='precomputed')
    sltry = pairwise_distances(random_samples, metric='correlation')
    X = dataset1.fit_transform(sltry)
    col_content = pd.DataFrame(X)

    return pd.json.dumps(col_content)


#this function is used to implement MDS euclidian (Random samples)
@app.route('/mdseuclR')
def mdseuclR():
    col_content = []
    #Dissimilarity measure to use:
# ‘euclidean’:Pairwise Euclidean distances between points in the dataset.
# ‘precomputed’:Pre-computed dissimilarities are passed directly to fit and fit_transform
    dataset1 = manifold.MDS(n_components=2, dissimilarity='precomputed')
     #Compute the distance matrix from a vector array X
    sltry = pairwise_distances(random_samples, metric='euclidean')
    X = dataset1.fit_transform(sltry)
    col_content = pd.DataFrame(X)

    print(pd.json.dumps(col_content))
    return pd.json.dumps(col_content)


#this function is used to implement MDS euclidian (adaptive samples)
@app.route('/mdseuclA')
def mdseuclA():
    col_content = []

# Dissimilarity measure to use:
# ‘euclidean’:Pairwise Euclidean distances between points in the dataset.
# ‘precomputed’:Pre-computed dissimilarities are passed directly to fit and fit_transform
    dataset1 = manifold.MDS(n_components=2, dissimilarity='precomputed')
    #Compute the distance matrix from a vector array X
    sltry = pairwise_distances(adaptive_samples[ftrs], metric='euclidean')
    X = dataset1.fit_transform(sltry)
    col_content = pd.DataFrame(X)
    return pd.json.dumps(col_content)


#this function is used to implement MDS correlation (adaptive samples)
@app.route('/mdscorrA')
def mdscorrA():
    col_content = []
        #Dissimilarity measure to use:
# ‘euclidean’:Pairwise Euclidean distances between points in the dataset.
# ‘precomputed’:Pre-computed dissimilarities are passed directly to fit and fit_transform
    dataset1 = manifold.MDS(n_components=2, dissimilarity='precomputed')
     #Compute the distance matrix from a vector array X
    sltry = pairwise_distances(adaptive_samples[ftrs], metric='correlation')
    #fit_transform
    X = dataset1.fit_transform(sltry)
    col_content = pd.DataFrame(X)

    return pd.json.dumps(col_content)


kelbow()
random_sampling()
generate(data,3)
squaredLoading=generate11(data,3)
print(squaredLoading)
print(sorted(squaredLoading))
print(list3)
list6=[]
# pca_filter_attr=squaredLoading
# print(pca_filter_attr)
# print(reversed(squaredLoading))

#this method is used for getting top 3 pca attributes indexes!!
sortlist=sorted(squaredLoading)
for i in sortlist:
    print (i)
    for j,k in zip(squaredLoading,range(len(squaredLoading))):
        if j==i:
            print(k)
            list6.append(k)
print(list6[::-1])  

pca_filter_attr=list6[::-1]
print(pca_filter_attr)
print("*")
# scatter_matrix_random()
pcaR()
# generate1(random_samples,3)

if __name__ == "__main__":
 
    app.run(debug=True)