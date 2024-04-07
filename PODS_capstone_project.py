# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:33:08 2023

@author: Kai-Hsuan Chan
"""
from sklearn.tree import plot_tree
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.special import expit
import seaborn as sns
from scipy.stats import logistic
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score





#seeding the rng
mySeed = 10078679
random.seed(mySeed)
np.random.seed(mySeed)

data = pd.read_csv("C:/Users/shayn/Downloads/spotify52kData.csv", delimiter = ',')
data.dropna(inplace=True)

#EDA

#display summary statistics
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("\nSummary Statistics:")
print(data.describe(include = 'all'))

#plot histogram of numerical features
print("\nUnivariate Analysis - Histograms:")
data.hist(bins=15, figsize=(15, 10))
plt.tight_layout()
plt.show()


#correlation matrix for numerical features
print("\nCorrelation Analysis:")
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

song_features = data[["duration", "danceability","energy", "loudness","speechiness","acousticness", "instrumentalness", "liveness","valence", "tempo"]]

song_features_np_array = song_features.values



#Question 1

num_features =  song_features_np_array.shape[1]
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))  # Adjust the figure size as needed
fig.suptitle('Distribution of Song Features')

# Flatten the 2D array of subplots to 1D
axes = axes.flatten()

# Loop through each column and create a histogram
for i in range(num_features):
    axes[i].hist(song_features_np_array[:, i], bins=30, edgecolor='black', color='skyblue')
    axes[i].set_title(song_features.columns[i])
    axes[i].set_xlabel('Values')
    axes[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add space for the title
plt.show()

#from the plots, the features danceability and tempo seem to be normally distibuted

descriptivesContainer = np.empty([num_features,5]) #Initialize as empty
descriptivesContainer[:] = np.NaN  #Filling them with nans to begin with

for ii in range(num_features):
    descriptivesContainer[ii,0] = np.mean(song_features_np_array[:,ii]) # Mean
    descriptivesContainer[ii,1] = np.median(song_features_np_array[:,ii]) # Mean
    descriptivesContainer[ii,2] = np.std(song_features_np_array[:,ii]) # SD
    descriptivesContainer[ii,3] = len(song_features_np_array[:,ii]) # n
    descriptivesContainer[ii,4] = descriptivesContainer[ii,1]/np.sqrt(descriptivesContainer[ii,2]) # SEM

# this seems to be true as the mean is similar to the median for danceability and tempo



#Question 2
duration = (data[["duration"]]).values.transpose()
popularity = data[["popularity"]].values.transpose()


# Create scatter plot
plt.scatter(duration, popularity, label='Scatter Plot')

# Add labels and title
plt.xlabel('duration')
plt.ylabel('popularity')
plt.title('duration-popularity plot')


# Show the plot
plt.show()

#there doesent seem to be a correlation between duration and popularity of a song from looking at the plot

#we will still do a pearson correaltion between the two variables just to be sure
correlation_coef, p_value = stats.pearsonr(data['duration'], data['popularity'])

print(correlation_coef, p_value)




#Question 3
#looking at the correlation matrix, there indeed is no correlation between the two variables
#we will first divied the data into explicit rated songs and non explicit songs
explicit = data[data["explicit"]==True]
non_explicit = data[data["explicit"]==False]

#we will first check if the data is normally distributed to decide on what test to use
data['popularity'].hist(bins=20)
plt.title('Histogram of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

#based on the histogram, there seems to be a spike at value 0, so we will be doing a nonparametric test
#the mann-whitney u test is a good test for this since it takes the median, which is more robust in this case


statistic, p_value = mannwhitneyu(explicit["popularity"], non_explicit["popularity"], alternative='greater')

print(f'Mann-Whitney U Statistic: {statistic}')
print(f'P-value: {p_value}')

# Check significance level
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in popularity between explicit and non-explicit songs.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in popularity between explicit and non-explicit songs.")


#the test is significant at the 0.05 significance level, so there is a difference
median_explicit = explicit["popularity"].median()
median_non_explicit = non_explicit["popularity"].median()

#we will also compare the median of explicited and non-explicited songs, to see which one is more popular
print(f"Median Popularity for Explicit Songs: {median_explicit}")
print(f"Median Popularity for Non-Explicit Songs: {median_non_explicit}")

if median_explicit > median_non_explicit:
    print("Explicit songs tend to be more popular.")
elif median_explicit < median_non_explicit:
    print("Non-Explicit songs tend to be more popular.")
else:
    print("There is no clear difference in median popularity between explicit and non-explicit songs.")

#based on the u test and the comparison of median, we conclude that explicited songs tend to be more popular

print()


#Question 4
#we will also use a similar approach to see if songs in major key are more popular than songs in minor key
major = data[data["mode"]==1]
minor = data[data["mode"]==0]
statistic, p_value = mannwhitneyu(minor["popularity"], major["popularity"], alternative='greater')

print(f'Mann-Whitney U Statistic: {statistic}')
print(f'P-value: {p_value}')

# Check significance level
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in popularity between songs in major and minor keys.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in popularity between songs in major and minor keys.")
    

median_major = major["popularity"].median()
median_minor = minor["popularity"].median()

#we will also compare the median of explicited and non-explicited songs, to see which one is more popular
print(f"Median Popularity for major Songs: {median_major}")
print(f"Median Popularity for minor: {median_minor}")

if median_major > median_minor:
    print("major songs tend to be more popular.")
elif median_major < median_minor:
    print("minor songs tend to be more popular.")
else:
    print("There is no clear difference in median popularity between major and minor songs.")
    
print()

#question 5
#we will first plot the data in a scatter plot
energy = data[["energy"]]
loudness = data[["loudness"]]
plt.scatter(energy, loudness)
plt.title("Scatter Plot of Energy-Loudness")
plt.xlabel("Energy")
plt.ylabel("Loudness")
plt.show()

#the data doesn't seem to be completely linear, however, it is monotonic, so we will use
#a spearman correlation

#next, we will check the corrleation between energy and loudness
# Calculate Spearman correlation
spearman_corr, p_value = spearmanr(energy, loudness)

print("Spearman correlation coefficient:", spearman_corr)
print("P-value:", p_value)
#we get a correlation coeffecient of 0.73 between energy and loudness
#based on this, we suspect that energy should be somewhat reflective of loudness

#I decided to do a simple linear regression on the data for further exploration
X_train, X_test, y_train, y_test = train_test_split(energy, loudness, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# Visualize the results
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='Linear Regression Model')
plt.xlabel('Energy')
plt.ylabel('Loudness')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()

#looking at the R-squared, we see that it is about 0.6,
#which means that energy explains 60% of the variance in loudness
#therefore, energy is indeed largely reflective of loudness

#question 6
#to answer this quesiton, i will do a linear regression for each of the 10 song features on popularity
#and compare the results



R2_results =  np.empty([num_features,1])
mse_results =  np.empty([num_features,1])
corr_results = np.empty([num_features,1])
for ii in range(num_features):
    X = song_features.iloc[:, ii].values.reshape(-1, 1)
    y = data[["popularity"]]
    
    corr = np.corrcoef(song_features.iloc[:, ii].values.transpose(), y.values.transpose())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create a linear regression model
    model = LinearRegression()
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    #add the reuslts to their respective containers
    R2_results[ii] = r2
    mse_results[ii] = mse
    corr_results[ii] = corr[0,1]

    
    
# Assuming 'R2_results' contains the R-squared values for each feature
feature_names = ["duration", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrument", "liveness", "valence", "tempo"]

# Create a dictionary to store feature names and their corresponding R-squared values
feature_r2_dict = {feature: r2 for feature, r2 in zip(feature_names, R2_results.flatten())}

# Sort features based on R-squared values in descending order
sorted_features = sorted(feature_r2_dict, key=feature_r2_dict.get, reverse=True)

# Plot the R-squared values in descending order
plt.bar(range(len(sorted_features)), [feature_r2_dict[feature] for feature in sorted_features], align='center')
plt.xticks(range(len(sorted_features)), sorted_features, rotation=45)
plt.xlabel('Features')
plt.ylabel('R-squared')
plt.title('R-squared Values for Each Feature (Descending Order)')
plt.show()

# instrumentalness has the highest R2 score of 0.022.
# however, this suggests that this feature alone isn't predictive of popularity
# since the R2 score is so low, it barealy explains any of the variance in popularity

print()

#question 7
#we will first do a multiple linear regression to see how it performs


popularity = data[["popularity"]]
song_features = data[["duration", "danceability","energy", "loudness","speechiness","acousticness", "instrumentalness", "liveness","valence", "tempo"]]




X = song_features.values
y = popularity.values.ravel()  # Flatten the target array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root mean Squared Error: {rmse}')

#I will try to normalize the rmse by converting it to a percentage of the range
#which will make it easier to interpret
y = popularity.values
target_range = np.max(y) - np.min(y)
normalized_rmse_range = (rmse / target_range) * 100
print(f'Normalied root mean Squared Error: {normalized_rmse_range}')

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

#the normalized root mean squared error is 21.1% of the range, and the r-squared is 0.04. This indicates
#that the model isn't performing well in predicting popularity, as it only accounts for 4 percent
#of the variance in popularity

print()

#question 8
# Extract the features from the DataFrame
X = song_features.to_numpy()

# Standardize the features (important for PCA)
zscored_song_features = stats.zscore(X)

pca = PCA().fit(zscored_song_features)
eigVals = pca.explained_variance_
loadings = pca.components_

rotatedData = pca.fit_transform(zscored_song_features) * -1 #multiple by -1 to adjust for polarity


varExplained = eigVals/sum(eigVals)*100

print("variance explained for each component in decreasing order:")
# Now let's display this for each factor:
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))


# What a scree plot is: A bar graph of the sorted Eigenvalues
numQuestions = 10
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='blue')
plt.plot([0,numQuestions],[1,1],color='red') 
plt.ylabel('Eigenvalue')
plt.show()


whichPrincipalComponent = 2 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show() # Show bar plot

#I choose to use the kaiser criterion to extract meaningful components, which yielded 3
#song_features = data[["duration", "danceability","energy", "loudness","speechiness","acousticness", "instrumentalness", "liveness","valence", "tempo"]]
# Plot the loadings matrix for each principal component in a single frame


# Plot the loadings matrix for the top 3 principal components
num_components_to_plot = 3

fig, axs = plt.subplots(1, num_components_to_plot, figsize=(5 * num_components_to_plot, 5), sharey=True)

for i in range(num_components_to_plot):
    axs[i].bar(x, loadings[i, :] * -1)
    axs[i].set_xlabel('Question')
    axs[i].set_ylabel(f'Loading (PC{i + 1})')
    axs[i].set_title(f'Loadings for Principal Component {i + 1}')

plt.show()

# Visualize the data using a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(rotatedData[:, 0], rotatedData[:, 1], rotatedData[:, 2], c='blue', marker='o')

ax.set_xlabel('Vitality')
ax.set_ylabel('Positivity')
ax.set_zlabel('How emotional?')
ax.set_title('Data Visualization in 3D using the First Three Principal Components')
ax.zaxis.labelpad=-3 
plt.show()


# Store our transformed data - the predictors - as x:
x = np.column_stack((rotatedData[:,0],rotatedData[:,1],rotatedData[:,2]))


numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters,1])*np.NaN # init container to store sums

# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii), n_init=10).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = sum(s) # take the sum
    print(ii)
 

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()


#based on the summed silluoette scores, the optimal k is 2, as it has the highest summed silhouette score
#now we have the optimal clustering, we will try and visualize the data in 3d

# kMeans:
k = 2

# Perform k-means clustering
kmeans = KMeans(n_clusters=k)
cluster_assignments = kmeans.fit_predict(x)

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot points with different colors for each cluster
for i in range(k):
    cluster_points = x[cluster_assignments == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {i + 1}')

# Set axis labels
ax.set_xlabel('Vitality')
ax.set_ylabel('positivity')
ax.set_zlabel('How emotional?')

# Set plot title
ax.set_title('K-Means Clustering in 3D Space')
# Add a legend
ax.legend()
ax.zaxis.labelpad=-3 
# Show the plot
plt.show()


#question 9
#since the outcome is binary (major or minor), I will use a logistic regression model

song_features = data[["duration", "danceability", "energy", "loudness", "speechiness","acousticness", "instrumentalness", "liveness","valence", "tempo"]]
# Calculate the correlation matrix
correlation_matrix = song_features.corr()


#before doing a logistic regression, i will first plot the data
plt.scatter(data[["valence"]].values,data[["mode"]].values,color='black', edgecolors="orange")
plt.xlabel('valence')
plt.ylabel('mode')
plt.show()
#there seems to be no discrepancy between data points that are major and data points that are minor
#so this suggests that valence wouldn't be a good predictor for major/minor

valence = data[["valence"]].values
# Target variable
target = data[["mode"]].values


X_train, X_test, y_train, y_test = train_test_split(valence, target, test_size=0.2)

y_train = y_train.ravel()
# Initialize the logistic regression model
model = LogisticRegression().fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Plot the logistic regression decision boundary and sigmoid function
x_values = np.linspace(-30, 30, 500).reshape(-1, 1)
linear_combination = x_values * model.coef_ + model.intercept_
y_probabilities = expit(linear_combination)

# Scatter plot of the data points
plt.scatter(X_test, y_test, color='black', label='Actual Data Points')
plt.plot(x_values, y_probabilities, color='red', linewidth=3, label='Logistic Regression Decision Boundary')

# Set labels and title
plt.xlabel('Valence')
plt.ylabel('Mode (0=minor, 1=major)')
plt.title('Logistic Regression and Sigmoid Function')

# Show the plot
plt.show()


#looking at the plot, the data points are all grouped in the middle of the sigmoid function
#this is eveidence that the prediction isn't really good
#we will use a confustion to quantify the performance


confusion_mat = confusion_matrix(y_test, predictions)
print(f'Confusion Matrix:\n{confusion_mat}')

#looking at the confusion matrix, there are 3951 false positives and 6449 true positive
#0 false negatives and 0 true negatives
#this is really bad as this suggests that the model predicted every outcome as "major"
#as the majority of the mode is "major"
#this means that valence isn't even close to being a good predictor for mode 

y_probabilities = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#the ROC curve is nearly identical to being a random guess, further supporting the claim 
#that valence isn't a good predictor for major/minor songs
auc_values = []

mode = data[["mode"]]
y = mode.values.ravel()

# Initialize an empty list to store AUC scores
auc_scores = []

# Loop through each feature in song_features
for feature in song_features.columns:
    # Extract the feature variable
    X_feature = song_features[[feature]].values

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)

    # Initialize logistic regression model
    model = LogisticRegression().fit(X_train, y_train)

    # Predict probabilities on the test set
    y_probabilities = model.predict_proba(X_test)[:, 1]

    # Compute AUC score
    auc_score = roc_auc_score(y_test, y_probabilities)

    # Append AUC score to the list
    auc_scores.append(auc_score)

# Plot histogram of AUC scores for each feature
plt.bar(range(1, 11, 1), auc_scores, color='blue')
plt.xlabel('Song Features')
plt.ylabel('AUC Score')
plt.title('AUC Scores for Each Song Feature in Logistic Regression')
plt.show()

#question 10

#before building the model, we will first transform the track_genre column variables into numeric values
#since they are string values, we cannot use the data. It needs preprocessing


# Assuming 'genre' is the name of your column
genres = data['track_genre']

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the genre column
numeric_genres = label_encoder.fit_transform(genres) + 1  # Adding 1 to start from 1

#we have now converted the genres to numeric values, ranging from 1 to 52
#so there are 52 types of genres in total

#in order to predict the genres, i decided to use a random forest classifier
#and we will use the 10 song features as a predictor

song_features = data[["duration", "danceability","energy", "loudness","speechiness","acousticness", "instrumentalness", "liveness","valence", "tempo"]]


#without train-test split
numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(song_features,numeric_genres) #bagging numTrees trees

# Use model to make predictions:
predictions = clf.predict(song_features) 
#accuracy: 0.8890384615384616

# Assess model accuracy:
modelAccuracy = accuracy_score(numeric_genres,predictions)
print('Random forest model accuracy:',modelAccuracy)


#with train-test split

X = data[["duration", "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]
y = numeric_genres

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(X_train, y_train)

# Use model to make predictions on the test set:
predictions = clf.predict(X_test)

# Assess model accuracy on the test set:
modelAccuracy = accuracy_score(y_test, predictions)
print('Random forest model accuracy on the test set:', modelAccuracy)

plt.figure(figsize=(20, 10))
plot_tree(clf.estimators_[0], filled=True, feature_names=song_features.columns, class_names=list(map(str, clf.classes_)), rounded=True)
plt.show()
