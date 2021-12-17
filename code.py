from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
#data = pd.read_csv("breast-cancer-wisconsin.data")
#data.columns = ["id", "ClumpThick", "UniSize", "UniShape", "MargAd", "SingEpiCelSize", "Bare Nuc", "BlandChr", "NormalNuc", "Mito", "Class"]
#data.to_csv("breastdata.csv", index=None, header=True)
data = pd.read_csv("breastdata.csv")
data.drop(['id'], inplace=True, axis=1) #gets rid of id column as not needed for analysis
data.replace('?', -99999, inplace = True) #replaces empty values
data["Class"] = data["Class"].map(lambda i: 1 if i ==4 else 0) #replaces malignant and benign to binary values
print(data)
X = np.array(data.drop(["Class"], axis =1))
Y = np.array(data["Class"])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) #data split into 80/20 for training and test
clf1 = KNeighborsClassifier(n_neighbors=7)
clf2 = RandomForestClassifier(n_estimators=500, max_depth =5)
estimators = [ ('KNN', clf1), ('Random Forest', clf2)]
clf3 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
"""
df = yf.download('^GSPC', start='2010-01-01')

df['returns'] = np.log(df.Close.pct_change() + 1)

#data frame manipulated and lags added as columns which are named
def lagsindataP(df, lags):
    lagname=[]
    for i in range(1, lags+1):
        df['Lag_'+str(i)] = df['returns'].shift(i)
        lagname.append('Lag_'+str(i))
        return lagname
  """  

labels = ['KNN', 'Random Forest', 'Stacking']
"""
param_grid1 = {
    'n_neighbors': range(1,14),
    'weights': ['uniform', 'distance'],
    }
param_grid2 = {
    'n_estimators': [100, 200, 300, 400, 500, 800, 12000],
    'max_depth': [3, 5, 10, None],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    }

knngrid = GridSearchCV(clf1, param_grid1, n_jobs=-1, cv=3)
rfcgrid = GridSearchCV(clf2, param_grid2, n_jobs=-1, cv=3)

labelsgrid = ['KNN', 'RFC', 'SKC']
for clfgrid, label in zip([knngrid, rfcgrid], labelsgrid):
    clfgrid = clfgrid.fit(X_train, Y_train)
    print(label + " best parameters:")
    print(clfgrid.best_params_)
"""  

for clf, label in zip([clf1, clf2, clf3], labels): #executes metric scores for all 3 classifiers
    clf = clf.fit(X_train, Y_train)
    cv = ShuffleSplit(n_splits=10, test_size=0.1) #data split into 10 foldss all representing 10%
    from sklearn.model_selection import cross_val_score
    print(label+":")
    accuracyscore = cross_val_score(clf, X, Y, cv=cv, scoring='accuracy')
    print("Cross fold validation accuracy scores:",accuracyscore)
    print("Cross fold validation accuracy mean:",accuracyscore.mean())
    print()
    precisionscore = cross_val_score(clf, X, Y, cv=cv, scoring='precision_macro')
    print("Cross fold validation precision scores:",precisionscore,)
    print("Cross fold validation precision mean:",precisionscore.mean())
    print()
    f1score=cross_val_score(clf, X, Y, cv=cv, scoring='f1_macro')
    print("Cross fold validation f1 scores:",f1score,)
    print("Cross fold validation f1 mean:",f1score.mean())
    print()
    recallscore=cross_val_score(clf, X, Y, cv=cv, scoring='recall_macro')
    print("Cross fold validation recall scores:",recallscore,)
    print("Cross fold validation recall mean:",recallscore.mean())
    print()
    print()