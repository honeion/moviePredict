import pandas as pd
import os
import numpy as np
# 데이터 가져오기 및 이상 데이터 제거
# modeling_data = pd.read_csv(os.getcwd() + '/../data/z_non_scaled_value_data_D0.csv', index_col=0)
# testing_data = pd.read_csv(os.getcwd() + '/../data/z_test_non_scaled_value_data_D0.csv', index_col=0)
modeling_data = pd.read_csv(os.getcwd() + '/../data/25_value_data_0_noD.csv', index_col=0)
testing_data = pd.read_csv(os.getcwd() + '/../data/26_test_data_0_noD.csv', index_col=0)
# testing_data = testing_data[testing_data["final_audience"] > 50000].reset_index(drop=True)
ALL = len(modeling_data)
print(ALL)
# 1점짜리 신예는 5점 ㄱ
def resizeValue(i):
    if i < 500000:
        return "00_50down"
    elif i >= 500000 and i < 1000000:
        return "01_50up 100down"
    elif i >= 1000000 and i < 2000000:
        return "02_100up 200down"
    elif i >= 2000000 and i < 3000000:
        return "03_200up 300down"
    elif i >= 3000000 and i < 4000000:
        return "04_300up 400down"
    elif i >= 4000000 and i < 5000000:
        return "05_400up 500down"
    elif i >= 5000000 and i < 6000000:
        return "06_500up 600down"
    elif i >= 6000000 and i < 7000000:
        return "07_600up 700down"
    elif i >= 7000000 and i < 8000000:
        return "08_700up 800down"
    elif i >= 8000000 and i < 9000000:
        return "09_800up 900down"
    elif i >= 9000000 and i < 10000000:
        return "10_900up 1000down"
    else:
        return "11_1000up"

from sklearn.metrics import mean_squared_error

def EvaluatingAccuracyWithError(model, predictions, train_d, train_t, test_d, test_t, real_target):  # 예측 정확도 평가
    print(predictions)
    print("-"*50)
    print(real_target)
    print("-" * 50)
    errors = abs(predictions - real_target)
    print(errors)
    # print(errors.count(0))
    count = 0
    for i in errors:
        if(i==0):
            count = count + 1
    return count
    # train_error, validation_error = calc_metrics(train_d, train_t, test_d, test_t, model)
    # print('\t Train Error {:10} : '.format(' '), round(train_error, 2))
    # print('\t Validation Error {:5} : '.format(' '), round(validation_error, 2))

    ### 성능 메트릭스 결정
    # accuracy = np.around(100 - (errors / target) * 100, 2) 아래와 같은 얘기
    # mape = 100 * (errors / real_target)  # 각각 다 나눈 것을 % 매긴 것 # Mean Absolute Percentage Error 이긴 함
    # accuracy = 100 - np.mean(mape)  # 에러의 평균을 100에서 뺀 것.
    #  np.around(100 - ((y - x) / a["final_audience"].get_values()) * 100, 2) 각각 에러를 100에서 뺀 것의 평균을 구하기 위함
    # print('\t 테스트 셋 예측 정확도 : ', round(accuracy, 2), '%.')  # 테스트 셋에 대한 예측률 평균
    # return [accuracy]#, train_error, validation_error]

movieCd = modeling_data["movieCd"]
modeling_data = modeling_data.drop(["movieCd"], axis=1)
modeling_data["final_audience"] = modeling_data["final_audience"].apply(resizeValue)
testing_data["final_audience"] = testing_data["final_audience"].apply(resizeValue)
testing_data = testing_data.drop(["movieCd"], axis=1)

dataset = modeling_data.ix[:, :-1]
test_dataset = testing_data.ix[:, :-1]
target = modeling_data.ix[:, -1]
test_target = testing_data.ix[:, -1]
dataset_list = list(dataset.columns)

resized_values = ["00_50down","01_50up 100down","02_100up 200down","03_200up 300down",
                  "04_300up 400down","05_400up 500down","06_500up 600down","07_600up 700down",
                  "08_700up 800down","09_800up 900down","10_900up 1000down","11_1000up"]
resized_values = resized_values[::-1]

#
from sklearn.model_selection import train_test_split

dataset_train, dataset_test, target_train, target_test = train_test_split(dataset, target, shuffle=True, test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier

print('Target 제외 DataSet Shape:', modeling_data.shape)
print('훈련 데이터셋 Shape:', dataset_train.shape)
print('훈련 타겟 Shape:', target_train.shape)
print('테스트 데이터셋 Shape:', dataset_test.shape)
print('테스트 타겟 Shape:', target_test.shape)

print(dataset_list)
categorized_number,categorized_index_names  = pd.factorize(target_train)
print("categorized_number:",categorized_number)

print("categorized_index_names:",categorized_index_names)
print("-"*50)

# real_target = pd.factorize(target_test)[0]

best_estimator = {'bootstrap': True, 'max_depth': 60, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 3000}

default_Random_Forest_clf =  RandomForestClassifier(
        n_estimators=best_estimator['n_estimators'],
        min_samples_split=best_estimator['min_samples_split'],
        min_samples_leaf=best_estimator['min_samples_leaf'],
        max_features=best_estimator['max_features'],
        max_depth=best_estimator['max_depth'],
        bootstrap=best_estimator['bootstrap'],
        random_state=42)

default_Random_Forest_clf.fit(dataset_train, categorized_number)
print(dataset_list)
print(default_Random_Forest_clf.feature_importances_)
predictions = default_Random_Forest_clf.predict(dataset_test) # [ 0  0  0  0  0  0  0 범주화된 번호 출력
predictionALL = default_Random_Forest_clf.predict(dataset) # [ 0  0  0  0  0  0  0 범주화된 번호 출력
predictionForTest = default_Random_Forest_clf.predict(test_dataset)
# predictionForTest = default_Random_Forest_clf.predict(test_dataset)

# print(predictionForTest)
# 1. 모든 애들 비교시
#   예측된 결과와 실제 결과의 차이를 수치화한다
#   단계의 차이
# 2. 분류방식
#   라벨 끼리 비교.
prediction = []     # 다시 라벨로 변경
for i in predictions:
    prediction.append(categorized_index_names[i])

prediction_all = []     # 다시 라벨로 변경
for i in predictionALL:
    prediction_all.append(categorized_index_names[i])

prediction_for_test = []     # 다시 라벨로 변경
for i in predictionForTest:
    prediction_for_test.append(categorized_index_names[i])

print(prediction_for_test)
# print(prediction_for_test)
def resultView(name):
    count = 0
    k = 0
    writer = pd.ExcelWriter(os.getcwd() + "/../data/z_test_output_" + name + "_clf_confusion_matrix.xlsx")
    DF = pd.DataFrame(columns=["index", "prediction", "realValue", "accuracy"])
    for i in target_test.index:
        real_value = target[target.index==i]
        # real_value.index.get_values(), prediction[k], real_value.get_values(), real_value.get_values() == prediction[k])
        value_list = [k, prediction[k], real_value.get_values(), real_value.get_values() == prediction[k]]
        if (real_value.get_values() == prediction[k]):
            count = count + 1
        DF.loc[len(DF)] = [value_list[0], value_list[1], value_list[2], value_list[3]]
        k = k+1
    print("count:", count)
    print("prediction_accuracy : ", round(count / len(target_test), 2) * 100, "%")
    # 오분류표
    confusion_matrix = pd.crosstab(target_test, np.array(prediction), rownames=['Movie Prediction'])
    confusion_matrix.to_excel(writer, "Sheet1")
    writer.save()
    DF.to_csv(os.getcwd() + "/../data/z_test_output_" + name + "_clf.csv", encoding="utf-8")
    # confusion_matrix = pd.crosstab(target_test, np.array(prediction), rownames=['Movie Prediction'])
    # print("count:",count)


# print(real_value.index.get_values(),  prediction[k], real_value.get_values(), real_value.get_values() == prediction[k])
# print(type(a.get_values()), type(prediction[i])) ndarray & str
def resultAllView(name): # RandomForest_D0
    count = 0
    writer = pd.ExcelWriter(os.getcwd()+"/../data/z_output_"+name+"_clf_confusion_matrix.xlsx")
    DF = pd.DataFrame(columns=["index", "prediction", "realValue", "accuracy"])
    for i in target.index:
        real_value = target[i]
        value_list = [i,prediction_all[i], real_value, real_value == prediction_all[i]]
        if (real_value == prediction_all[i]):
            count = count + 1
        DF.loc[len(DF)] = [value_list[0], value_list[1], value_list[2], value_list[3]]
    print("count:", count)
    print("prediction_accuracy : ", round(count/ALL,2)*100,"%")
    # 오분류표
    confusion_matrix = pd.crosstab(target, np.array(prediction_all), rownames=['Movie Prediction'])
    confusion_matrix.to_excel(writer, "Sheet1")
    writer.save()
    DF.to_csv(os.getcwd() + "/../data/z_output_" + name + "_clf.csv", encoding="utf-8")

def resultTestView(name): # RandomForest_D0
    count = 0
    writer = pd.ExcelWriter(os.getcwd()+"/../data/z_test_output_"+name+"_clf_confusion_matrix.xlsx")
    DF = pd.DataFrame(columns=["index", "prediction", "realValue", "accuracy"])
    for i in test_target.index:
        real_value = test_target[i]
        value_list = [i,prediction_for_test[i], real_value, real_value == prediction_for_test[i]]
        if (real_value == prediction_for_test[i]):
            count = count + 1
        DF.loc[len(DF)] = [value_list[0], value_list[1], value_list[2], value_list[3]]
    print("count:", count)
    print("prediction_accuracy : ", round(count/len(test_target),2)*100,"%")
    # 오분류표
    confusion_matrix = pd.crosstab(test_target, np.array(prediction_for_test), rownames=['Movie Prediction'])
    confusion_matrix.to_excel(writer, "Sheet1")
    writer.save()
    DF.to_csv(os.getcwd() + "/../data/z_test_output_" + name + "_clf.csv", encoding="utf-8")

def visualizationTree(name, predictions, feature_list):  # 트리 생성
    # 트리와 독립변수들의 중요도를 체크
    # 트리 시각화
    from sklearn.tree import export_graphviz
    import pydot
    # os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz/'  # 필수, 경로!
    # os.environ["PATH"] += os.pathsep + "C:/Users/KSWLab2/AppData/Local/Programs/Python/Python36 - 32/Lib/graphviz-2.38/"#/windows/bin"
    os.environ["PATH"] += os.pathsep +"C:/Users/KSWLab2/AppData/Local/Programs/Python/Python36 - 32/Lib/graphviz - 2.38/release/bin"
    tree = predictions.estimators_[0]
    export_graphviz(tree, out_file=name + '.dot', feature_names=feature_list, filled=True,
                    node_ids=True, rounded=True, precision=1)
    (graph,) = pydot.graph_from_dot_file(name + '.dot')
    # graph = pydot_ng.graph_from_dot_file(name + '.dot')
    graph.write_png(name + '.png')
print("-"*80)
resultView("RandomForest_D0")
print("-"*80)
# print(resultAllView("RandomForest_D0"))

# print("-"*80)
# resultTestView()
# print("-"*80)
# visualizationTree("yaya34",default_Random_Forest_clf,dataset_list)
# 각 영화에는 [1274] ['01_50up 100down'] 00_50down 이런값 넣어주고 맞췄다 틀렸다 저런 범주로 포함시키고 예측한것
# 전체 정확도는 컨퓨전 매트릭스랑 라쏘, 랜포 변수 중요도하는 부분에 같이 넣어주면 되겠다
# 적당한 설명이랑 read me  페이지정도 있어야 하겠다.

#일단 테스트셋 따로 전체 셋 따로 해보자.

# 배우랑 감독 점수는 보류
# 스코어링 하는것도 보류 // 점수 매기는 방식(50만,100만 이런거)

# 전체 점수 평가 기준 변경 - 편차를 늘림
# ---> 과적합이 늘었고, 큰 변화가 없었음
# 시간관계상 기존의 방식을 고수하기로 함
