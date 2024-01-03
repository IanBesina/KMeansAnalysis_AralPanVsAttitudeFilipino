import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read dataset
dataset = pd.read_csv('/home/whatever/Downloads/AP.csv')

# Do KMeans and create dataset clusters
# I decided that the number of cluster will be 4 instead of 5, as there is no respondent who got below 75 
# in Araling Panlipunan in both Don Bosco Cebu and Bacolod City NHS STEM Class
# Although I believe that age may not be of material significance as there is not much age differences
km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(dataset[['ethnicity', 'attitude', 'writtenfil', 'spokenfil', 'grade']])
dataset['cluster'] = y_predicted
print(dataset)
datasetcluster1 = dataset[dataset.cluster == 0]
datasetcluster2 = dataset[dataset.cluster == 1]
datasetcluster3 = dataset[dataset.cluster == 2]
datasetcluster4 = dataset[dataset.cluster == 3]

# Visualize clusters using scatter plot
# If my comprehension of KClustering is correct, this should be able to include other factors/columns in clustering
# However, Pyplot will limit its visualization into two axis, which temporarily be assigned to "attitude" vs. "grade"
# In the actual reports for the inheriting Professional Learning Community (PLC), plots displaying "ability in written Filipino"
# and "ability in spoken FIlipino" will also be shown.

# plt.scatter(datasetcluster1.writtenfil, datasetcluster1.grade, color='red')  
# plt.scatter(datasetcluster1.spokenfil, datasetcluster1.grade, color='red')  
plt.scatter(datasetcluster1.attitude, datasetcluster1.grade, color='red')
plt.scatter(datasetcluster2.attitude, datasetcluster2.grade, color='violet')
plt.scatter(datasetcluster3.attitude, datasetcluster3.grade, color='blue')
plt.scatter(datasetcluster4.attitude, datasetcluster4.grade, color='green')
plt.xlabel('Attitude Towards Other Filipino Culture Esp Tagalog')
plt.ylabel('Grade in Araling Panlipunan')

# Test model
test_ethnicity = 5
test_attitude = 3
test_writtenfil = 3
test_spokenfil = 3
test_grade = 3
test_cluster = km.predict([[test_ethnicity, test_attitude, test_writtenfil, test_spokenfil, test_grade]])
# plt.scatter(test_writtenfil, test_grade, color='yellow')
# plt.scatter(test_spokenenfil, test_grade, color='yellow')
plt.scatter(test_attitude, test_grade, color='yellow')
plt.show()

print('CLUSTER:', test_cluster)

def showElbow():
    k_range = range(1, 10)
    sse = []
    for k in k_range:
        test_km = KMeans(n_clusters=k)
        test_km.fit(dataset[['attitude', 'grade']])
        sse.append(test_km.inertia_)
    plt.plot(k_range, sse)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.show()

# showElbow()