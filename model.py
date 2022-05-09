import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the csv file
df = pd.read_csv("iris.csv")

print(df.head())

# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# print(y.value_counts(normalize=False))
# print(y_train.value_counts(normalize=False))
# print(y_test.value_counts(normalize=False))

pipe = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
print(score)

# Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# print(X_train)
#
# # Instantiate the model
# classifier = RandomForestClassifier()

# Fit the model
# classifier.fit(X_train, y_train)

# Make pickle file of our model
with open('model.pickle', 'wb') as handle:
    pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)
