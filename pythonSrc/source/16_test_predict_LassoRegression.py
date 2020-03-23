#-*-coding:utf-8-*-
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import scipy as sp
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split   #cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# alpha 0일차 1000, 7일차 2000, 14일차 630 28일차 40

# ALL = 1800
# modeling_data = pd.read_csv(os.getcwd() +'/../data/12_value_data_D0_1800.csv',index_col =0)
modeling_data = pd.read_csv(os.getcwd() +'/../data/24_non_scaled_value_data_D0.csv',index_col =0)
# testing_data = pd.read_csv(os.getcwd() +'/../data/30_test_value_data_D0.csv',index_col =0)
testing_data = pd.read_csv(os.getcwd() +'/../data/test_non_scaled_value_data_D0.csv',index_col =0)
# testing_data = testing_data[testing_data["final_audience"]>50000].reset_index(drop=True)
print(len(testing_data))
ALL = len(testing_data)

movieCd = modeling_data["movieCd"]
modeling_data = modeling_data.drop(["movieCd"],axis=1)
testing_data = testing_data.drop(["movieCd"],axis=1)
dataset = modeling_data.ix[:,:-1]
test_dataset = testing_data.ix[:,:-1]
target = modeling_data.ix[:,-1]
test_target = testing_data.ix[:,-1]

def view_result(model, number, dataset, target):  # 결과 보여주는 함수
    a = target[target.index == number]  # 1행 전체 라인 -> target 1개 뿐
    b = model.predict(dataset)[number]  # (data 1행 라인)
    x = min(a.get_values(), b)
    y = max(a.get_values(), b)
    allPredict = np.around(100 - ((y - x) / a.get_values()) * 100, 2)  # y

    index = a.index.get_values()
    prediction = np.around(b, 0)
    realValue = a.get_values()
    accuracy = allPredict
    print("번호 : ", index, "예측값 : ", prediction,
          "실제값 : ", realValue, "예측율 : ", allPredict, "%")
    return allPredict, index, prediction, realValue, accuracy

def resultAllPredict(model, datalist,targetlist, name):  # 전체 영화 예측 결과
    allPredict = 0
    count = 0
    DF = pd.DataFrame(columns=["index", "prediction", "realValue", "accuracy"])
    for i in range(0, ALL):
        a = view_result(model, i, datalist, targetlist)
        b = a[0]
        count = count + 1
        allPredict += b
        DF.loc[len(DF)] = [a[1], a[2], a[3], a[4]]

    print('예측된 전체 데이터 셋 정확도 : ', allPredict / count)
    print('사용 갯수 : ', count)
    DF.to_csv(os.getcwd()+"/../data/99_test_output_"+name+".csv", encoding="utf-8") #99는 결과

def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return mse

def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse

def calc_metrics(X_train, y_train, X_test, y_test, model):
    '''fits model and returns the RMSE for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score

import warnings                     #covergence warning 무시코드
warnings.filterwarnings("ignore", category=ConvergenceWarning)

dataset_train, dataset_test, target_train, target_test = train_test_split(dataset, target, shuffle=True, test_size=0.2, train_size=0.8)

bestAlpha28 = 40
bestAlpha14 = 603
bestAlpha7 = 2000
bestAlpha0 = 1000
#0
lassoRegression = Lasso(alpha=bestAlpha0, normalize=True, positive=True, fit_intercept=True, random_state=77)

























































































































































































# #7
# lassoRegression = Lasso(alpha=bestAlpha7, normalize=True, positive=True, fit_intercept=True, random_state=77)
# #14
# lassoRegression = Lasso(alpha=bestAlpha14, normalize=True, fit_intercept=True, random_state=77)
#28
# lassoRegression = Lasso(alpha=bestAlpha28, normalize=True, fit_intercept=True, random_state=77)

lassoRegression.fit(dataset_train,target_train)
# print("-"*70)
# print("Train Set Prediction")
# count = 0
# summary = 0
# for index in target_train.index:
#     real = target[target.index == index].get_values()
#     predict = lassoRegression.predict(dataset_train)[count]
#     x = min(real, predict)
#     y = max(real, predict)
#     allPredict = np.around(100 - ((y - x) / real) * 100, 2)
#     print("번호 :", str(index).ljust(5), "예측값 :", str(np.around(predict, 0)).ljust(10), "실제값 :", str(real).ljust(10),
#           "예측율 :", allPredict, "%")
#     summary += allPredict
#     count += 1
#
# print("평균 :", summary/len(target_train), "%")
# print(len(target_train))
# print("사용한 특성의 수: {}".format(np.sum(lassoRegression.coef_ != 0)))
#
# print("-"*70)
# print("Test Set Prediction")
# count = 0
# summary = 0
# for index in target_test.index:
#     real = target[target.index == index].get_values()
#     predict = lassoRegression.predict(dataset_test)[count]
#     x = min(real, predict)
#     y = max(real, predict)
#     allPredict = np.around(100 - ((y - x) / real) * 100, 2)
#     print("번호 :", str(index).ljust(5), "예측값 :", str(np.around(predict, 0)).ljust(10), "실제값 :", str(real).ljust(10),
#           "예측율 :", allPredict, "%")
#     summary += allPredict
#     count += 1
#
# print("평균 :", summary/len(target_test), "%")
# print(len(target_test))
# print("사용한 특성의 수: {}".format(np.sum(lassoRegression.coef_ != 0)))
# print("-"*70)

count = 0
lassoRegression_col_list = []
lassoRegression_col_index = []
remove_column_list=[]
for i in range(0,len(lassoRegression.coef_)):
    print('[Variable] {:20} : [Coefficient] {}'.format(dataset.columns[i],lassoRegression.coef_[i]))
    if lassoRegression.coef_[i] == 0.0:
        count += 1
        remove_column_list.append(dataset.columns[i])
    else:
        lassoRegression_col_list.append(dataset.columns[i])
        lassoRegression_col_index.append(i)
print(len(dataset.columns),count,len(dataset.columns)-count)
print("-"*50)
print(lassoRegression_col_list)
print(dataset.values)
print("-"*50)
print('[Intercept]:',lassoRegression.intercept_)
print('[Parameters]:',lassoRegression.get_params())

dataset = dataset.drop(remove_column_list, axis=1)
print(len(dataset.columns))
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

mergedDF = pd.concat([dataset,target],axis=1)
formula_str = "final_audience ~ " + " + ".join(lassoRegression_col_list)
y,X = dmatrices(formula_str,data=mergedDF,return_type="dataframe")
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(dataset.values, i) for i in range(0,len(lassoRegression_col_list))]
vif["features"] = lassoRegression_col_list
print(vif.round(1))
# vif.to_csv(os.getcwd()+'/../data/20_lasso_regression_D0_vif.csv', encoding='utf-8')

resultAllPredict(lassoRegression, test_dataset,test_target, "LassoRegression_D0")
