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


modeling_data = pd.read_csv(os.getcwd() +'/../data/11_value_data_D7.csv',index_col =0)
modeling_data=modeling_data.drop("movieCd",axis=1)
dataset = modeling_data.ix[:, :-1]
target = modeling_data.ix[:, -1]
ALL = len(modeling_data)

def view_result(model, number, dataset):  # 결과 보여주는 함수
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

def resultAllPredict(model, datalist, name):  # 전체 영화 예측 결과
    allPredict = 0
    count = 0
    DF = pd.DataFrame(columns=["index", "prediction", "realValue", "accuracy"])
    for i in range(0, ALL):
        a = view_result(model, i, datalist)
        # if a > 0:
        b = a[0]
        count = count + 1
        allPredict += b
        DF.loc[len(DF)] = [a[1], a[2], a[3], a[4]]

    print('예측된 전체 데이터 셋 정확도 : ', allPredict / count)
    print('사용 갯수 : ', count)
    # DF.to_csv(os.getcwd()+"/../data/99_output_"+name+".csv", encoding="utf-8") #99는 결과


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

'''
# alphas = [0.01,0.1,1,20,30,40,45,50,60,70,100,1000] #알파 9선택
# bestalpha = 40

# coef_print=pd.DataFrame()
# validation_score_print=pd.DataFrame()
# coef_print["features"] = data.columns
# validation_score_print["alpha"] = alphas
# val_score_temp=[]
#cross validation을 이용한 알파값 선정
# for alpha in alphas:
#     # lasso = Lasso(alpha=alpha, fit_intercept=True, tol=0.0001, max_iter=1000, positive=True, random_state=77)
#
#     # lasso = Lasso(alpha=alpha, normalize=True, fit_intercept=True, random_state=77)
#     lasso.fit(data,target)
#     column_name = 'Alpha = %f'%alpha
#     scores = cross_val_score(lasso, data, target, cv=10)
#     sum1 = 0
#     j=0
#     for i in scores:
#         sum1 += i
#     print("알파: ",alpha," : ",sum1 / len(scores))
#     coef_print[column_name] = lasso.coef_
#     val_score_temp.append(sum1 / len(scores))
#     print("사용특성 수 : {}".format(np.sum(lasso.coef_ != 0)))
# print(val_score_temp)
# validation_score_print["val_score"] = val_score_temp


# coef_print.to_csv(os.getcwd()+'/coef_print.csv',encoding='utf-8')
# validation_score_print.to_csv(os.getcwd()+'/validation_score_print.csv',encoding='utf-8')



# # K-fold로 알파 찾기!!!
# K = 10
# kf = KFold(n_splits=K, shuffle=True, random_state=42)
# 
# fold_alpha_print = ['alpha','train_error','val_error']
# fold_alph_temp=[]
# for alpha in alphas:
#     train_errors = []
#     validation_errors = []
#     for train_index, val_index in kf.split(data, target):
#         # split data
#         X_train, X_val = data.iloc[train_index], data.iloc[val_index]
#         # X_train, X_val = data[train_index], data[val_index]
#         y_train, y_val = target[train_index], target[val_index]
# 
#         # instantiate model
# 
#         # lasso = Lasso(alpha=alpha, normalize=True, fit_intercept=True, random_state=77)
#         # lasso = Lasso(alpha=alpha, tol=0.007, max_iter=1000, fit_intercept=True, normalize=True)
#         lasso.fit(data,target)
# 
#         # calculate errors
#         train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, lasso)
# 
#         # append to appropriate list
#         train_errors.append(train_error)
#         validation_errors.append(val_error)
# 
#     # generate report
# 
#     print('alpha: {:6} | mean(train_error): {:7} | mean(val_error): {}'.
#           format(alpha,
#                  round(np.mean(train_errors), 4),
#                  round(np.mean(validation_errors), 4)))
#     print("사용특성 수 : {}".format(np.sum(lasso.coef_ != 0)))
# 
#     fold_alph_temp.append([alpha, round(np.mean(train_errors), 4), round(np.mean(validation_errors), 4)])
# 
# df1 = pd.DataFrame(fold_alph_temp,columns=fold_alpha_print)
# df1.to_csv(os.getcwd()+'/fold_alpha_print.csv',encoding='utf-8')

'''

dataset_train, dataset_test, target_train, target_test = train_test_split(dataset, target, shuffle=True, test_size=0.2, train_size=0.8)
# 위에서 나온 알파값을 이용한 Lasso 회귀분석 실행!!!
# La1.to_csv(os.getcwd()+'/La1.csv', encoding='utf-8')
bestAlpha28 = 40
bestAlpha14 = 603
bestAlpha7 = 2000
bestAlpha0 = 1000
#0
# lassoRegression = Lasso(alpha=bestAlpha0, normalize=True, positive=True, fit_intercept=True, random_state=77)
# #7
lassoRegression = Lasso(alpha=bestAlpha7, normalize=True, positive=True, fit_intercept=True, random_state=77)
# #14
# lassoRegression = Lasso(alpha=bestAlpha14, normalize=True, fit_intercept=True, random_state=77)
#28
# lassoRegression = Lasso(alpha=bestAlpha28, normalize=True, fit_intercept=True, random_state=77)

lassoRegression.fit(dataset_train,target_train)
print("-"*70)
print("Train Set Prediction")
count = 0
summary = 0
for index in target_train.index:
    real = target[target.index == index].get_values()
    predict = lassoRegression.predict(dataset_train)[count]
    x = min(real, predict)
    y = max(real, predict)
    allPredict = np.around(100 - ((y - x) / real) * 100, 2)
    print("번호 :", str(index).ljust(5), "예측값 :", str(np.around(predict, 0)).ljust(10), "실제값 :", str(real).ljust(10),
          "예측율 :", allPredict, "%")
    summary += allPredict
    count += 1

print("평균 :", summary/len(target_train), "%")
print(len(target_train))
print("사용한 특성의 수: {}".format(np.sum(lassoRegression.coef_ != 0)))

print("-"*70)
print("Test Set Prediction")
count = 0
summary = 0
for index in target_test.index:
    real = target[target.index == index].get_values()
    predict = lassoRegression.predict(dataset_test)[count]
    x = min(real, predict)
    y = max(real, predict)
    allPredict = np.around(100 - ((y - x) / real) * 100, 2)
    print("번호 :", str(index).ljust(5), "예측값 :", str(np.around(predict, 0)).ljust(10), "실제값 :", str(real).ljust(10),
          "예측율 :", allPredict, "%")
    summary += allPredict
    count += 1

print("평균 :", summary/len(target_test), "%")
print(len(target_test))
print("사용한 특성의 수: {}".format(np.sum(lassoRegression.coef_ != 0)))
print("-"*70)


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
# resultAllPredict(lassoRegression, dataset, "LassoRegression_D7")

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
# vif.to_csv(os.getcwd()+'/../data/19_lasso_regression_D7_vif.csv', encoding='utf-8')

import matplotlib.pyplot as plt
xvalues = range(0,len(lassoRegression.coef_))
# plt.bar(y_pred,y_test,color='b')
plt.figure(figsize=(14, 10))
plt.grid(color='black', linestyle='--', )
# plt.subplot(131)
plt.bar(xvalues, lassoRegression.coef_,  color='red', width = 0.5)
plt.xticks(xvalues, modeling_data.columns, rotation=45, fontsize=10)
plt.title("Lasso_D7 Coefficient", fontsize=15, verticalalignment='bottom')
plt.show()


# def visualizationImportance(name, dataset_columns, feature_list):
#     xvalues = range(0, len(feature_list))
#     # plt.bar(y_pred,y_test,color='b')
#     plt.figure(figsize=(17, 12))
#     plt.grid(color='black', linestyle='--', )
#     # plt.subplot(131)
#     plt.bar(xvalues, feature_list, color='black', width=0.5)
#     plt.xticks(xvalues, dataset_columns, rotation=60, fontsize=10)
#     plt.title("RandomForest_D28 " + name, fontsize=15, verticalalignment='bottom')
#     plt.show()