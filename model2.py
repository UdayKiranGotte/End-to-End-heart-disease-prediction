import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('heart.csv')

X = df.drop('target',axis=1)
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.3,random_state=50)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

pickle.dump(classifier,open('model2.pkl','wb'))