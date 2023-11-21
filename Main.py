import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 

spam = pd.read_csv('spam.csv') # Read the CSV file 'spam.csv' and create a DataFrame assigned to the variable 'spam'

z = spam['EmailText'] #  Extracts the 'EmailText' column from the DataFrame spam and assigns it to the variable z
y = spam["Label"] # Extracts the 'Label' column from the DataFrame spam and assigns it to the variable y
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2) # Uses the train_test_split function from scikit-learn to split the data into training and testing sets. 

cv = CountVectorizer() # tokenizing,  it counts the number of occurrences of words and saves it to cv
features = cv.fit_transform(z_train)  