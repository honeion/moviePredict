import pandas as pd
import os
import numpy as np
# 데이터 가져오기 및 이상 데이터 제거
modeling_data = pd.read_csv(os.getcwd() + '/../data/RFC_test/21_non_scaled_value_data_D28.csv', index_col=0)
# modeling_data = modeling_data.drop("movieCd", axis=1)
# print(modeling_data)
movieCd = modeling_data["movieCd"]
modeling_data = modeling_data.drop(["movieCd"], axis=1)
dataset = modeling_data.ix[:, :-1]
target = modeling_data.ix[:, -1]
dataset_list = list(dataset.columns)
def score_change(i):
    if i < 100000:
        return "00_10down"
    elif i >= 100000 and i < 200000:
        return "01_10up 20down"
    elif i >= 200000 and i < 300000:
        return "02_20up 30down"
    elif i >= 300000 and i < 400000:
        return "03_30up 40down"
    elif i >= 400000 and i < 500000:
        return "04_40up 50down"
    elif i >= 500000 and i < 1000000:
        return "05_50up 100down"
    elif i >= 1000000 and i < 2000000:
        return "06_100up 200down"

    elif i >= 2000000 and i < 3000000:
        return "07_200up 300down"
    elif i >= 3000000 and i < 4000000:
        return "08_300up 400down"
    elif i >= 4000000 and i < 5000000:
        return "09_400up 500down"
    elif i >= 5000000 and i < 6000000:
        return "10_500up 600down"
    elif i >= 6000000 and i < 7000000:
        return "11_600up 700down"
    elif i >= 7000000 and i < 8000000:
        return "12_700up 800down"
    elif i >= 8000000 and i < 9000000:
        return "13_800up 900down"
    elif i >= 9000000 and i < 10000000:
        return "14_900up 1000down"
    else:
        return "15_1000up"
target_names = ["00_10down","01_10up 20down","02_20up 30down","03_30up 40down","04_40up 50down",
                "05_50up 100down","06_100up 200down","07_200up 300down","08_300up 400down",
                "09_400up 500down","10_500up 600down","11_600up 700down","12_700up 800down",
                "13_800up 900down","14_900up 1000down","15_1000up"]
target_names = target_names[::-1]

from sklearn.ensemble import RandomForestClassifier

modeling_data["final_audience"] = modeling_data["final_audience"].apply(score_change)
modeling_data['is_train'] = np.random.uniform(0, 1, len(modeling_data)) <= .75

train, test = modeling_data[modeling_data['is_train']==True], modeling_data[modeling_data['is_train']==False]

# print('Number of observations in the training data:', len(train))
# print('Number of observations in the test data:',len(test))

features = modeling_data.columns[:-2]
# print(features)
# print(train[features])
y, z = pd.factorize(train['final_audience'])
print("y:",y)
#
print("z:",z)
print("-"*50)
#
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y)

testSet = test['final_audience']
print(test[features])

result = clf.predict(test[features])
print(testSet)
print("-"*50)
print(result)
print("-"*50)
preds = []
for i in result:
    preds.append(target_names[i])
print(preds)
print("-"*50)

a = pd.crosstab(test['final_audience'],np.array(preds), rownames=['Actual Species'], colnames=['Predicted Species'])
print(a)

print(list(zip(train[features], clf.feature_importances_)))
#
# importances_list = list(default_Random_ForestR.feature_importances_)
#     feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(dataset_list, importances_list)]
#     feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) # 2번째 값 기준으로 정렬하고 역순으로 재배치 해야 큰값 우선으로 정렬됨
#
#     [print('변수: {:20} 중요도: {}'.format(*pair)) for pair in feature_importances];