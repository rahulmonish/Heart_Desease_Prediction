#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:19:02 2020

@author: rahul
"""

import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


df= pd.read_csv('heart.csv')
df.isnull().sum()


correlation_plot= df.corr(method='pearson')
y= df['target']
#sns.pairplot(df, hue='target',diag_kind='hist')

column_list=[]
for i in df.columns:
    if i!= 'target':
        if float(df['target'].corr(df[i])) > 0.0:
            #print(float(df['target'].corr(df[i])))
            column_list.append(i)



df= df[column_list]
#df= df.drop(['target'], axis=1)
x= df


bestfeatures = SelectKBest(score_func=chi2, k=13)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()




x_train,x_test,y_train,y_test= train_test_split(x, y, test_size= .3)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1',
                                                                  'principal component 2'])
principalDf['target']= y
# =============================================================================
# principalDf['target']= principalDf['target'].astype(str)
# for i in range(0, len(principalDf)):
#     if principalDf['target'].at[i]=='0':
#         principalDf['target'].at[i]='red'
#     else:
#         principalDf['target'].at[i]='green'
#
# import matplotlib.pyplot as plt
# fig = plt.scatter(
#                  x=principalDf['principal component 1'],
#                  y=principalDf["principal component 2"],
#                  c= principalDf['target'])
# fig.show()
# =============================================================================
#principalDf['target']= y

    #sns.pairplot(principalDf, hue='target', diag_kind='hist')

principalDf= principalDf.values
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



model= Sequential()
sgd = keras.optimizers.Adam(lr=0.1)

model.add(Dense(3,input_dim=4,kernel_initializer='he_normal', activation='relu'))
model.add(Dense(4,kernel_initializer='he_normal', activation='relu'))
#model.add(Dropout(.2))
model.add(Dense(1,activation='sigmoid' ))
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(principalDf,y, epochs= 50)




#XGBoost Classifier
model= xg_reg = xgb.XGBClassifier(subsample= 1.0,
                                 min_child_weight= 10,
                                 learning_rate= 0.1,
                                 gamma= 1.5,
                                 booster= 'gbtree',
                                 colsample_bytree= 1.0)
model.fit(x_train,y_train)
y_predict= model.predict(x_test)

#RandomForest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
y_predict= model.predict(x_test)

accuracy= accuracy_score(y_predict, y_test)


from sklearn.model_selection import cross_val_score
score= cross_val_score(model, x, y, cv=20, scoring='accuracy')
print(score.mean())

