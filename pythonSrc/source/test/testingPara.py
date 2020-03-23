

import pandas as pd
import os

import numpy as np

# modeling_data = pd.read_csv(os.getcwd() +'/../data/12_value_data_D0_1800.csv',index_col =0)
modeling_data = pd.read_csv(os.getcwd() + '/../data/25_value_data_0_noD.csv', index_col=0)
# testing_data = pd.read_csv(os.getcwd() +'/../data/30_test_value_data_D0.csv',index_col =0)
# testing_data = pd.read_csv(os.getcwd() + '/../data/26_test_data_0_noD.csv', index_col=0)
# testing_data = testing_data[testing_data["final_audience"] > 50000].reset_index(drop=True)
# print(len(testing_data))
# ALL = len(testing_data)

openDt_score_set = pd.read_csv(os.getcwd() + '/../data/score/openDt_score.csv',index_col =0)          # 개봉날짜
prdtYear_score_set = pd.read_csv(os.getcwd() + '/../data/score/prdtYear_score.csv',index_col =0)      # 제작연도
nation_score_set = pd.read_csv(os.getcwd() + '/../data/score/nation_score.csv',index_col =0)          # 국가
genre_score_set = pd.read_csv(os.getcwd() + '/../data/score/genre_score.csv',index_col =0)            # 장르
showTm_score_set = pd.read_csv(os.getcwd() + '/../data/score/showTm_score.csv',index_col =0)          # 쇼타임
watchGrade_score_set = pd.read_csv(os.getcwd() + '/../data/score/watchGrade_score.csv',index_col =0)  # 등급
director_score_set = pd.read_csv(os.getcwd() + '/../data/score/director_score.csv',index_col =0)      # 감독
actor_score_set = pd.read_csv(os.getcwd() + '/../data/score/actor_score.csv',index_col =0)            # 배우
distributor_score_set = pd.read_csv(os.getcwd() + '/../data/score/distributor_score.csv',index_col =0)# 배급사

def month_change(i):#str(data_set["openDt"][0])[4:6]
    if i == "01":
        return 'jan'
    elif i == '02':
        return 'feb'
    elif i == '03':
        return 'mar'
    elif i == '04':
        return 'apr'
    elif i == '05':
        return 'may'
    elif i == '06':
        return 'jun'
    elif i == '07':
        return 'jul'
    elif i == '08':
        return 'aug'
    elif i == '09':
        return 'sep'
    elif i == '10':
        return 'oct'
    elif i == '11':
        return 'nov'
    else:
        return 'dec'

def makeTest(movieCd,openDt):
    testing_data = pd.DataFrame(columns=["movieCd", "openDt_score"])
        # , "prdtYear_score", "repNationNm_score", "repGenreNm_score",
        #                              "showTm_score", "watchGradeNm_score", "distributor_score", "actor_score",
        #                              "starScore", "userCount", "previous_screen", "previous_show","previous_audience",
        #                              "final_audience"  # 최종 관객수
        #                              ])
    testing_data.loc[len(testing_data)] = [
        movieCd,
        openDt,
    ]
    return testing_data
df = makeTest('2012136114','20180910')
print(df)

print(openDt_score_set)

#
#
# movieCd = modeling_data["movieCd"]
# modeling_data = modeling_data.drop(["movieCd"], axis=1)
# testing_data = testing_data.drop(["movieCd"], axis=1)
# dataset = modeling_data.ix[:, :-1]
# test_dataset = testing_data.ix[:, :-1]
# target = modeling_data.ix[:, -1]
# test_target = testing_data.ix[:, -1]
# dataset_list = list(dataset.columns)
#
# # 훈련 및 테스트 셋
# from sklearn.model_selection import train_test_split
#
# dataset_train, dataset_test, target_train, target_test = train_test_split(dataset, target, shuffle=True, test_size=0.2, random_state = 42)
#
# ### TRAIN MODEL
# from sklearn import tree
# from sklearn.ensemble import RandomForestRegressor
# import matplotlib.pyplot as plt
#
# def view_result(model, number, dataset, target):  # 결과 보여주는 함수
#     a = target[target.index == number]  # 1행 전체 라인 -> target 1개 뿐
#     b = model.predict(dataset)[number]  # (data 1행 라인)
#     x = min(a.get_values(), b)
#     y = max(a.get_values(), b)
#     allPredict = np.around(100 - ((y - x) / a.get_values()) * 100, 2)  # y
#
#     index = a.index.get_values()
#     prediction = np.around(b, 0)
#     realValue = a.get_values()
#     accuracy = allPredict
#     print("번호 : ", index, "예측값 : ", prediction,
#           "실제값 : ", realValue, "예측율 : ", allPredict, "%")
#     return allPredict, index, prediction, realValue, accuracy
#
# def resultAllPredict(model, datalist, targetlist, name):  # 전체 영화 예측 결과
#     allPredict = 0
#     count = 0
#     DF = pd.DataFrame(columns=["index", "prediction", "realValue", "accuracy"])
#     for i in range(0, ALL):
#         a = view_result(model, i, datalist, targetlist)
#         b = a[0]
#         count = count + 1
#         allPredict += b
#         DF.loc[len(DF)] = [a[1], a[2], a[3], a[4]]
#
#     print('예측된 전체 데이터 셋 정확도 : ', allPredict / count)
#     print('사용 갯수 : ', count)
#     DF.to_csv(os.getcwd() + "/../data/99_test_output_" + name + ".csv", encoding="utf-8")  # 99는 결과
#
# from sklearn.metrics import mean_squared_error
#
# def calc_train_error(X_train, y_train, model):
#     predictions = model.predict(X_train)
#     mse = mean_squared_error(y_train, predictions)
#     rmse = np.sqrt(mse)
#     return mse
#
# def calc_validation_error(X_test, y_test, model):
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#     return mse
#
# def calc_metrics(X_train, y_train, X_test, y_test, model):
#     model.fit(X_train, y_train)
#     train_error = calc_train_error(X_train, y_train, model)
#     validation_error = calc_validation_error(X_test, y_test, model)
#     return train_error, validation_error
#
# def EvaluatingAccuracyWithError(model, predictions, train_d, train_t, test_d, test_t, real_target):  # 예측 정확도 평가
#     errors = abs(predictions - real_target)
#     train_error, validation_error = calc_metrics(train_d, train_t, test_d, test_t, model)
#     print('\t Train Error {:10} : '.format(' '), round(train_error, 2))
#     print('\t Validation Error {:5} : '.format(' '), round(validation_error, 2))
#
#     ### 성능 메트릭스 결정
#     # accuracy = np.around(100 - (errors / target) * 100, 2) 아래와 같은 얘기
#     mape = 100 * (errors / real_target)  # 각각 다 나눈 것을 % 매긴 것 # Mean Absolute Percentage Error 이긴 함
#     accuracy = 100 - np.mean(mape)  # 에러의 평균을 100에서 뺀 것.
#     #  np.around(100 - ((y - x) / a["final_audience"].get_values()) * 100, 2) 각각 에러를 100에서 뺀 것의 평균을 구하기 위함
#     print('\t 테스트 셋 예측 정확도 : ', round(accuracy, 2), '%.')  # 테스트 셋에 대한 예측률 평균
#     return [accuracy, train_error, validation_error]
#
# def visualizationTree(name, predictions, feature_list): # 트리 생성
#     # 트리와 독립변수들의 중요도를 체크
#     # 트리 시각화
#     from sklearn.tree import export_graphviz
#     import pydot
#     os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz/' # 필수, 경로!
#
#     tree = predictions.estimators_[1]
#     export_graphviz(tree, out_file = name+'.dot', feature_names = feature_list, filled=True,
#                     node_ids=True, rounded = True, precision = 1)
#     (graph, ) = pydot.graph_from_dot_file(name+'.dot')
#     graph.write_png(name+'.png')
# #
# def visualizationImportance(name,dataset_columns,feature_list):
#     xvalues = range(0, len(feature_list))
#     # plt.bar(y_pred,y_test,color='b')
#     plt.figure(figsize=(20, 7))
#     plt.grid(color='black', linestyle='--', )
#     # plt.subplot(131)
#     plt.bar(xvalues, feature_list, color='black')
#     plt.xticks(xvalues, dataset_columns, rotation=45, fontsize=6.5)
#     plt.title(name)
#     plt.show()
#
# print("-" * 50)
# print("default_Random_ForestR result - features(15)")
# default_Random_ForestR = RandomForestRegressor(random_state = 42)
# default_Random_ForestR.fit(dataset_train, target_train)
# resultAllPredict(default_Random_ForestR, test_dataset, test_target, "RandomForest_D0")
# print("-"*50)
# #
# ## Variable Importances 변수 중요성
# importances_list = list(default_Random_ForestR.feature_importances_)
# feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(dataset_list, importances_list)]
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) # 2번째 값 기준으로 정렬하고 역순으로 재배치 해야 큰값 우선으로 정렬됨
#
# print("-" * 50)
# print("default_Random_ForestR_Most_Importance_95 result - features(6)")
# default_Random_ForestR_Importance_features_list = [] #주요변수 담은 리스트공간
# for i in range(0,6):
#     default_Random_ForestR_Importance_features_list.append(feature_importances[i][0])
# important_indices = [] # 실제 인덱스 담을 리스트공간
# for i in default_Random_ForestR_Importance_features_list:
#     important_indices.append(dataset_list.index(i))
#
# train_important_dataset = dataset_train.iloc[:, important_indices]
# test_important_dataset = dataset_test.iloc[:, important_indices]
# all_important_dataset = dataset.iloc[:, important_indices]
# important_test_dataset = test_dataset.iloc[:, important_indices]
# # 줄인 피쳐로 훈련 시키고, 줄인 피쳐 가진 테스트 셋으로 예측
# default_Random_ForestR_Most_Importance_95 = RandomForestRegressor(random_state=42)
# default_Random_ForestR_Most_Importance_95.fit(train_important_dataset, target_train)
# resultAllPredict(default_Random_ForestR_Most_Importance_95, important_test_dataset, test_target,"RandomForest_D0")
#
# default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators = {'bootstrap': True, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
# grid_Search_default_RF_95 = RandomForestRegressor(
#     n_estimators=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['n_estimators'],
#     min_samples_split=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['min_samples_split'],
#     min_samples_leaf=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['min_samples_leaf'],
#     max_features=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['max_features'],
#     max_depth=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['max_depth'],
#     bootstrap=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['bootstrap'],
#     random_state=42)
#
# grid_Search_default_RF_95.fit(train_important_dataset, target_train)
#
# print("-" * 50)
#
# print("grid_Search_default_RF_95 result - features(6)")
# resultAllPredict(grid_Search_default_RF_95, important_test_dataset, test_target,"RandomForest_D0")
#
