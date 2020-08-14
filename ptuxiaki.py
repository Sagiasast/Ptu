import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from matplotlib import style
style.use("ggplot")

events= pd.read_csv("C:/Users/nikos/PycharmProjects/ptuxiaki/all_events_final1_fscore_wind.csv", sep = ";") #,na_values=["None"] )
events =pd.DataFrame(events)
events_=events.replace(',','.' , regex= True).astype(float)

#Data Preprocessing


X = events.drop( "MagnM" , axis= 1 )
X= preprocessing.scale(X)
y = events["MagnM"]
y= preprocessing.scale(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20 ,random_state=110)

lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)#labed gia binary
y_test_encoded = lab_enc.fit_transform(y_test)

#f-score

N_features = 8
selector = SelectKBest(f_classif, k=N_features)               # k is the number of features
selector.fit(X_train ,y_train)
scores=selector.scores_
#print(scores)
sharps = ["Vmean","Vmax","Bmax","VmBm","Bdo_Bup","DV","Ndo_Nup","Thita"]
plt.clf()
order = np.argsort(scores)
orderedsharps = [sharps[i] for i in order]
y_pos2 = np.arange(8)
plt.barh(y_pos2, sorted(scores/np.nanmax(scores)), align='center')
plt.ylim((-1, 8))
plt.xlim(0,1)
plt.yticks(y_pos2, orderedsharps)
plt.xlabel('Συσχέτιση', fontsize=15)
plt.title('Κανονικοποιημένο F-Score', fontsize=15)
fig = plt.gcf()
fig.set_size_inches(8,10)
plt.show()

# finding hyperplane
param = {'kernel' : ['poly','rbf','linear','sigmoid'],
         'C' : [1,5,10],'degree' : [3,8],
        'coef0' : [0.01,10,0.5],
       'gamma' : ('auto','scale')}
modelsvr = SVR()
grids = GridSearchCV(modelsvr,param,cv=5)
grids.fit(X_train,y_train)
grids.best_params_
print(grids.best_params_)



# Training the Algorithm
regressor = SVR(kernel='poly',degree=3,gamma='auto',coef0=0.01,C=5)
regressor.fit(X_train,y_train)

#predictions
y_pred = regressor.predict(X_test)
dff = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(dff)
print(y_test)
print(regressor.intercept_)
#print({'Parameter': sharps, 'corellation0': regressor.coef_})
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print(regressor.intercept_)

#{'C': 5, 'coef0': 0.01, 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf'}
#{'C': 1, 'coef0': 0.01, 'degree': 3, 'gamma': 'scale'}
