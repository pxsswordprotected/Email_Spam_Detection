import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 

spam = pd.read_csv('spam.csv') # Read the CSV file 'spam.csv' and create a DataFrame assigned to the variable 'spam'

z = spam['EmailText'] #  Extracts the 'EmailText' column from the DataFrame spam and assigns it to the variable z
y = spam["Label"] # Extracts the 'Label' column from the DataFrame spam and assigns it to the variable y
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2) # Uses the train_test_split function from scikit-learn to split the data into training and testing sets. 

cv = CountVectorizer() # tokenizing,  it counts the number of occurrences of words and saves it to cv
features = cv.fit_transform(z_train)  #randomly assigns a number to each word. It counts the number of occurrences of each word, then saves it to cv

model = svm.SVC() # assigns svm.SVC() to the model
model.fit(features,y_train) # trains the model with features and y_train. Then, it checks the prediction against the y_train label and adjusts its parameters until it reaches the highest possible accuracy.

features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))