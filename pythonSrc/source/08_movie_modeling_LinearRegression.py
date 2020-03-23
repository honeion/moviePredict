#-*-coding:utf-8-*-

import pandas as pd
import os
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
import numpy as np

# 기존의 라쏘, 랜덤포레스트는 value data를 그대로 이용한 것이 아니고, 다중회귀로 정제된 데이터를 사용해버렸던 것
# 문제가 있을 수 있다..!
modeling_data = pd.read_csv(os.getcwd() +'/../data/12_value_data_D0.csv',index_col =0)
# modeling_data = modeling_data[modeling_data["final_audience"]>1000000].reset_index(drop=True)
# modeling_data.to_csv(os.getcwd()+"/../data/12_value_data_D0.csv",encoding="utf-8")
ALL = len(modeling_data) # 1800
# movieCd = modeling_data["movieCd"]
# modeling_data=modeling_data.drop("movieCd",axis=1)
# print(modeling_data.dtypes)

# for id in X.columns:#range(len(X.columns)):
#     print(X[id][0],type(X[id][0]))
movieCd = modeling_data["movieCd"]
modeling_data = modeling_data.drop(["movieCd"],axis=1)
dataset = modeling_data.ix[:,:-1]
target = modeling_data.ix[:,-1]

from sklearn.model_selection import train_test_split #cross validation


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
    DF.to_csv(os.getcwd()+"/../data/99_output_"+name+".csv", encoding="utf-8") #99는 결과

from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

lr_model = LinearRegression()
cv = KFold(5)
scores = np.zeros(5)

for i, (train_index, test_index) in enumerate(cv.split(dataset)):
    X_train = dataset.ix[train_index]
    y_train = target.ix[train_index]
    X_test = dataset.ix[test_index]
    y_test = target.ix[test_index]
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    scores[i] = r2_score(y_test, y_pred)

print("[CV Score(R2)] :",scores)
from sklearn.model_selection import cross_val_score

print(-cross_val_score(lr_model, dataset, target, scoring="neg_mean_squared_error",cv=cv))#, scoring="r2", cv=cv))
# print(-cross_val_score(model, dataset, target, scoring="r2", cv=cv))
for i in range(0,len(lr_model.coef_)):
    print('[Variable] {:20} : [Coefficient] {}'.format(dataset.columns[i],lr_model.coef_[i]))
print('[Intercept]:',lr_model.intercept_)
print('[Parameters]:',lr_model.get_params())

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
movie_columns = list(modeling_data.columns)[:-1]
# print(movie_columns)
formula_str = "final_audience ~ " + " + ".join(movie_columns)
y,X = dmatrices(formula_str,data=modeling_data,return_type="dataframe")
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(dataset.values, i) for i in range(dataset.shape[1])]
vif["features"] = dataset.columns
print(vif.round(1))
# vif.to_csv(os.getcwd()+'/../data/16_linear_regression_D7_vif.csv', encoding='utf-8')

# resultAllPredict(lr_model, dataset, "LinearRegression_D7")
#
import matplotlib.pyplot as plt
xvalues = range(0,len(lr_model.coef_))
# plt.bar(y_pred,y_test,color='b')
plt.figure(figsize=(25, 7))
plt.grid(color='black', linestyle='--', )
# plt.subplot(131)
plt.bar(xvalues, lr_model.coef_,  color='blue')
plt.xticks(xvalues, modeling_data.columns, rotation=45, fontsize=6.5)
plt.show()

# 밸리데이션만 추가했는데, 결과는 낫배드한데
# vif(분산팽창요인) 확인해보니 문제가 있더라 그래서 라쏘를 사용해서 문제를 해결했다. //라쏘해서 줄인 부분에서 vif 한번 해야함