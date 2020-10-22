from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import svm
import pickle

import pandas as pd
mydata = pd.read_csv(r"C:\Users\Aditya\Desktop\heart.csv")

x = mydata.iloc[:,:-1].values
y = mydata.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.neighbors import KNeighborsClassifier

teacher  = KNeighborsClassifier()
learner = teacher.fit(x_train, y_train)
pickle.dump(learner,open('learner_KNN.pkl','wb'))

log_reg = LogisticRegression()
logistic = log_reg.fit(x_train,y_train)
pickle.dump(logistic,open('model_logistic.pkl','wb'))

teacher3 = LinearRegression()
linear = teacher3.fit(x_train,y_train)
pickle.dump(linear,open('model_linear.pkl','wb'))

clf = svm.SVC()
support = clf.fit(x_train,y_train)
pickle.dump(support,open('model_SVM.pkl','wb'))


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
x_train, y_train = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
tree = clf.fit(x_train, y_train)
pickle.dump(support,open('model_tree.pkl','wb'))




print("Done!")


