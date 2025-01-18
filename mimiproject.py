import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# taking data from csv-file
raw_data = pd.read_csv(r'C:/Users/pr094/Downloads/data.csv')

# replace null values
data = raw_data.where((pd.notnull(raw_data)),'')

# rows in dataframe
data.head()

# print number of rows and columns
data.shape

# Check unique values in Category column
print("Unique Categories:", data['Category'].unique())

# Useing LabelEncoder for category  encoding
labelencoder = LabelEncoder()
data['Category'] = labelencoder.fit_transform(data['Category'])

# print encoded categories
print("Encoded Categories:", data['Category'].unique())

# data separating
X = data['Message']
Y = data['Category']

# split data
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3, random_state=4)

# print shapes
print("\nTotal data shape:", X.shape)
print("Training data shape:", trainX.shape)
print("Testing data shape:", testX.shape)

# transform data to feature vectors.
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# convert text to features
trainX_features = feature_extraction.fit_transform(trainX)
testX_features = feature_extraction.transform(testX)

# create and train the model
model = LogisticRegression(max_iter=1000)

# model training.
model.fit(trainX_features, trainY)

# prediction on training data
predicting_on_traindata = model.predict(trainX_features)
accuracy_training= accuracy_score(trainY, predicting_on_traindata)
print('\nAccuracy on training data', accuracy_training)

# prediction on test data
prediction_on_testdata = model.predict(testX_features)
accuracy_test = accuracy_score(testY, prediction_on_testdata)
print('Accuracy on test data', accuracy_test)

# classification report
print('\nClassification Report')
print(classification_report(testY, prediction_on_testdata))

# confusion matrix
print('\nConfusion Matrix')
print(confusion_matrix(testY, prediction_on_testdata))

input = ["Did you hear about the new ""Divorce Barbie""? It comes with all of Ken's stuff!"]

# convert text to feature vectors
input_features = feature_extraction.transform(input)

# prediction
prediction = model.predict(input_features)
print('\nTest Input Prediction:', prediction)

if prediction[0] == labelencoder.transform(['ham'])[0]:
    print('Ham mail')
else:
    print('Spam mail')
