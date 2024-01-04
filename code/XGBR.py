from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel(r"./data.xlsx")

Xtrain = np.asarray(data.iloc[:,5:])
Ytrain = np.asarray(data.iloc[:,2])

score_5cv_all = []
for i in np.arange(0.01, 0.5, 0.01):
    rfc = XGBR(learning_rate=i)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print(
      "best_5cv_score:{}".format(score_max_5cv),
      "lr_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
n_lr_5cv = np.arange(0.01,0.5,0.01)[score_5cv_all.index(score_max_5cv)]
print(n_lr_5cv)


score_5cv_all = []
for i in range(0, 200, 1):
    rfc =XGBR(learning_rate=n_lr_5cv
              ,random_state=i)
    score_5cv =cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print(
      "best_5cv_score{}".format(score_max_5cv),
      "random_5cv:{}".format(score_5cv_all.index(score_max_5cv)))

random_state_5cv = range(0, 200)[score_5cv_all.index(max(score_5cv_all))]
print(random_state_5cv)



score_5cv_all = []
for i in range(1, 400, 1):
    rfc = XGBR(n_estimators=i,
               learning_rate=n_lr_5cv,
        random_state=random_state_5cv)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print(
      "best_5cv_score{}".format(score_max_5cv),
      "n_est_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
n_est_5cv = range(1,400)[score_5cv_all.index(score_max_5cv)]
print(n_est_5cv)

score_5cv_all = []
for i in range(1, 300, 1):
    rfc = XGBR(n_estimators=n_est_5cv,
               learning_rate=n_lr_5cv
                                , random_state=random_state_5cv
                                , max_depth=i)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    CV_predictions = cross_val_predict(rfc, Xtrain, Ytrain, cv=5)
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print(
      "best_5cv_score{}".format(score_max_5cv),
      "max_depth_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_depth_5cv = range(1,300)[score_5cv_all.index(score_max_5cv)]
print(max_depth_5cv )

score_5cv_all = []
for i in np.arange(0,5,0.05):
    rfc = XGBR(n_estimators=n_est_5cv,
               learning_rate=n_lr_5cv
                                , random_state=random_state_5cv
                                , max_depth=max_depth_5cv
                                ,gamma= i)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    CV_predictions = cross_val_predict(rfc, Xtrain, Ytrain, cv=5)
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print(
      "best_5cv_score{}".format(score_max_5cv),
      "gamma_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_gamma_5cv =  np.arange(0,5,0.05)[score_5cv_all.index(score_max_5cv)]
print(max_gamma_5cv)

score_5cv_all = []
for i in np.arange(0,5,0.05):
    rfc = XGBR(n_estimators=n_est_5cv,
               learning_rate=n_lr_5cv
                                , random_state=random_state_5cv
                                , max_depth=max_depth_5cv
                                , gamma=max_gamma_5cv 
                                , alpha=i)
    score_5cv = cross_val_score(rfc, Xtrain, Ytrain, cv=5).mean()
    CV_predictions = cross_val_predict(rfc, Xtrain, Ytrain, cv=5)
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print(
      "best_5cv_score{}".format(score_max_5cv),
      "alpha_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_alpha_5cv =  np.arange(0,5,0.05)[score_5cv_all.index(score_max_5cv)]
print(max_alpha_5cv)

XGB = XGBR(learning_rate=n_lr_5cv,n_estimators=n_est_5cv,random_state=random_state_5cv,max_depth=max_depth_5cv,gamma =max_gamma_5cv,alpha = max_alpha_5cv)
CV_score = cross_val_score(XGB, Xtrain, Ytrain, cv=5).mean()
CV_predictions = cross_val_predict(XGB, Xtrain, Ytrain, cv=5)
rmse = np.sqrt(mean_squared_error(Ytrain,CV_predictions))
print("5cv:",CV_score)
print("RMSE_5CV",rmse)
expvspred_5cv = {'Exp': Ytrain, 'Pred':CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel('./XGBoost_5fcv_predictions.xlsx')
