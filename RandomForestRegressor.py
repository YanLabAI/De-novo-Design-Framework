from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
from scipy.stats import norm, skew 
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap


data = pd.read_excel(r"./data.xlsx")
x = data.iloc[:,5:]
scaler =StandardScaler()
scaler.fit(x)
standard_data =scaler.transform(x)
standard_data_out =np.asarray(standard_data)
pd.DataFrame(standard_data_out).to_excel('standard.xlsx')


data = pd.read_excel(r"./standard.xlsx")
'''
Model optimization
'''


Xtrain = np.asarray(data.iloc[:,5:])
Ytrain = np.asarray(data.iloc[:,2])
score_5cv_all = []
for i in range(0, 200, 1):
    rfc =RandomForestRegressor(random_state=i)
    score_5cv =cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print("best_5cv_score:{}".format(score_max_5cv),
      "random_5cv:{}".format(score_5cv_all.index(score_max_5cv)))

random_state_5cv = range(0, 200)[score_5cv_all.index(max(score_5cv_all))]
print(random_state_5cv)


score_5cv_all = []
for i in range(1, 400, 1):
    rfc = RandomForestRegressor(n_estimators=i,
        random_state=random_state_5cv)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("best_5cv_score:{}".format(score_max_5cv),
      "n_est_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
n_est_5cv = range(1,400)[score_5cv_all.index(score_max_5cv)]
print(n_est_5cv)
score_test_all = []
score_5cv_all = []
for i in range(1, 300, 1):
    rfc = RandomForestRegressor(n_estimators=n_est_5cv
                                , random_state=random_state_5cv
                                , max_depth=i)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print(
      "best_5cv_score:{}".format(score_max_5cv),
      "max_depth_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_depth_5cv = range(1,300)[score_5cv_all.index(score_max_5cv)]
print(max_depth_5cv )

rfc = RandomForestRegressor(n_estimators=n_est_5cv,random_state=random_state_5cv,max_depth=max_depth_5cv)
CV_score = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
CV_predictions = cross_val_predict(rfc, Xtrain, Ytrain, cv=5)
rmse = np.sqrt(mean_squared_error(Ytrain,CV_predictions))
print("5cv:",CV_score)
print("rmse_5CV",rmse)
expvspred_5cv = {'Exp': Ytrain, 'Pred':CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel('./Random_forest_5fcv_predictions.xlsx')

plt.rcParams['axes.unicode_minus']=False

Xtrain = data.iloc[:,5:]

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
sns.set()
sns.set_style('white', {'font.sans-serif':['simhei','Arial']})

explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(Xtrain)

plt.subplots(figsize=(15,15),dpi=1080,facecolor='w')

plt.xlabel("Shap value")

shap.summary_plot(shap_values, Xtrain, plot_type="bar",show=False)
plt.savefig("feature importance.png",dpi=600,bbox_inches = 'tight')