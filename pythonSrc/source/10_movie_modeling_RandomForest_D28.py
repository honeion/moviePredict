#-*- coding:utf-8 -*-

if __name__ == '__main__':
    from joblib import Parallel, delayed
    import multiprocessing
    import pandas as pd
    import os

    # 데이터 가져오기 및 이상 데이터 제거
    modeling_data = pd.read_csv(os.getcwd() + '/../data/21_non_scaled_value_data_D28.csv', index_col=0)
    modeling_data = modeling_data.drop("movieCd", axis=1)

    print('Target 제외 DataSet Shape:', modeling_data.shape)
    # Target 제외 DataSet Shape: (1800, 98) 28일
    # Target 제외 DataSet Shape: (1800, 58) 14일 -무작위 서칭, 무작위서칭에서 랭크를 매겨, 랭크기반 파라미터로 그리드 서칭을 함, 그리드서칭으로 나온 베스트 파라미터로 다시 돌린다. 이걸로 결과를 보는거
    # Target 제외 DataSet Shape: (500, 37)   7일 -무작위 서칭, 무작위서칭에서 랭크를 매겨, 랭크기반 파라미터로 그리드 서칭을 함, 그리드서칭으로 나온 베스트 파라미터로 다시 돌린다. 이걸로 결과를 보는거
    # Target 제외 DataSet Shape: (500, 16)   0일 -무작위 서칭, 무작위서칭에서 랭크를 매겨, 랭크기반 파라미터로 그리드 서칭을 함, 그리드서칭으로 나온 베스트 파라미터로 다시 돌린다. 이걸로 결과를 보는거
    # 타겟과 데이터셋 구분
    import numpy as np

    dataset = modeling_data.ix[:, :-1]
    target = modeling_data.ix[:, -1]
    dataset_list = list(dataset.columns)

    # 훈련 및 테스트 셋
    from sklearn.model_selection import train_test_split
    train_dataset, test_dataset, train_target, test_target = train_test_split(dataset, target, test_size = 0.2, random_state = 42)

    print('훈련 데이터셋 Shape:', train_dataset.shape)
    print('훈련 타겟 Shape:', train_target.shape)
    print('테스트 데이터셋 Shape:', test_dataset.shape)
    print('테스트 타겟 Shape:', test_target.shape)
    # 훈련 데이터셋 Shape: (1440, 97)
    # 훈련 타겟 Shape: (1440,)
    # 테스트 데이터셋 Shape: (360, 97)
    # 테스트 타겟 Shape: (360,)

    ### TRAIN MODEL
    from sklearn import tree
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    ALL = 1800


    def view_result(model, number,dataset): # 결과 보여주는 함수
        a = target[target.index == number]  # 1행 전체 라인 -> target 1개 뿐
        b = model.predict(dataset)[number]  # (data 1행 라인)
        x = min(a.get_values(), b)
        y = max(a.get_values(), b)
        allPredict = np.around(100 - ((y - x) / a.get_values()) * 100, 2)  # y

        index = a.index.get_values()
        prediction = np.around(b, 0)
        realValue = a.get_values()
        accuracy = allPredict
        print("번호 : ",index , "예측값 : ",prediction ,
              "실제값 : ",realValue, "예측율 : ", allPredict, "%")
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
        DF.to_csv(os.getcwd() + "/../data/99_output_" + name + ".csv", encoding="utf-8")  # 99는 결과

    from sklearn.metrics import mean_squared_error

    def calc_train_error(X_train, y_train, model):
        predictions = model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        rmse = np.sqrt(mse)
        return mse

    def calc_validation_error(X_test, y_test, model):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        return mse

    def calc_metrics(X_train, y_train, X_test, y_test, model):
        model.fit(X_train, y_train)
        train_error = calc_train_error(X_train, y_train, model)
        validation_error = calc_validation_error(X_test, y_test, model)
        return train_error, validation_error

    def EvaluatingAccuracyWithError(model, predictions, train_d, train_t, test_d, test_t, real_target):  # 예측 정확도 평가
        errors = abs(predictions - real_target)
        train_error, validation_error = calc_metrics(train_d, train_t, test_d, test_t, model)
        print('\t Train Error {:10} : '.format(' '), round(train_error, 2))
        print('\t Validation Error {:5} : '.format(' '), round(validation_error, 2))

        ### 성능 메트릭스 결정
        # accuracy = np.around(100 - (errors / target) * 100, 2) 아래와 같은 얘기
        mape = 100 * (errors / real_target)  # 각각 다 나눈 것을 % 매긴 것 # Mean Absolute Percentage Error 이긴 함
        accuracy = 100 - np.mean(mape)  # 에러의 평균을 100에서 뺀 것.
        #  np.around(100 - ((y - x) / a["final_audience"].get_values()) * 100, 2) 각각 에러를 100에서 뺀 것의 평균을 구하기 위함
        print('\t 테스트 셋 예측 정확도 : ', round(accuracy, 2), '%.')  # 테스트 셋에 대한 예측률 평균
        return [accuracy, train_error, validation_error]

    def visualizationTree(name, predictions, feature_list): # 트리 생성
        # 트리와 독립변수들의 중요도를 체크
        # 트리 시각화
        from sklearn.tree import export_graphviz
        import pydot
        os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz/' # 필수, 경로!

        tree = predictions.estimators_[1]
        export_graphviz(tree, out_file = name+'.dot', feature_names = feature_list, filled=True,
                        node_ids=True, rounded = True, precision = 1)
        (graph, ) = pydot.graph_from_dot_file(name+'.dot')
        graph.write_png(name+'.png')
    #
    def visualizationImportance(name,dataset_columns,feature_list):
        xvalues = range(0, len(feature_list))
        # plt.bar(y_pred,y_test,color='b')
        plt.figure(figsize=(34, 11))
        plt.grid(color='black', linestyle='--', )
        # plt.subplot(131)
        plt.bar(xvalues, feature_list, color='blue',width=0.5)
        plt.xticks(xvalues, dataset_columns, rotation=45, fontsize=10)
        plt.title("RandomForest_D28 "+name, fontsize=15, verticalalignment='bottom')
        plt.show()
    # 기본조건으로 시작
    '''
     default 값
         n_estimators = 10, criterion = mse(mean squared errer), max_features = "auto"
         max_depth = None, min_samples_split = 2, min_samples_leaf = 1
         min_weight_fraction_leaf = 0, max_leaf_nodes = None, min_impurity_decrease = 0
         bootstrap = True, oob_score = False, n_jobs = 1, verbose = 0, warm_start = False
         rf = RandomForestRegressor(n_estimators = 1000, random_state = 42,oob_score=True)
    '''
    # 이게 1 세트 -> 모델 선언, 피팅, 예측, 정확도 검증, 전체 결과 확인, 트리 시각화
    print("-" * 50, "\n Default - 전체 98개 변수 사용")
    default_Random_ForestR = RandomForestRegressor(random_state = 42)
    default_Random_ForestR.fit(train_dataset, train_target)
    predictions = default_Random_ForestR.predict(test_dataset)
    default_Random_ForestR_AccuracyWithError = EvaluatingAccuracyWithError(default_Random_ForestR, predictions,
                                                                           train_dataset, train_target,
                                                                           test_dataset, test_target, test_target)
    default_Random_ForestR_result = {'model': 'default',
                                     'accuracy': default_Random_ForestR_AccuracyWithError[0],
                                     'train_error': default_Random_ForestR_AccuracyWithError[1],
                                     'validation_error': default_Random_ForestR_AccuracyWithError[2],
                                     'n_trees': default_Random_ForestR.get_params()['n_estimators'],
                                     'n_features': train_dataset.shape[1]}
    # visualizationTree("default_tree", default_Random_ForestR, dataset_list)

    print("-"*50)
    # resultAllPredict(default_Random_ForestR,dataset,"RandomForest_D0")
    ## Variable Importances 변수 중요성
    importances_list = list(default_Random_ForestR.feature_importances_)
    feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(dataset_list, importances_list)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True) # 2번째 값 기준으로 정렬하고 역순으로 재배치 해야 큰값 우선으로 정렬됨

    [print('변수: {:20} 중요도: {}'.format(*pair)) for pair in feature_importances]; # 튜플값을 따로 출력하는 방법 pprint보다 깨끗함
    # visualizationImportance("default_importance",train_dataset.columns, importances_list)
    # # ---------------------------------------------------------------------------------------------------------------1.2
    # # 무작위 서칭
    # # 최상의 하이퍼파라미터를 찾기 위해 랜덤 그리드 사용
    # # 무작위로 파라미터 검사하고 5 fold 교차검증 함.
    # # 우선적으로 파라미터 조정전에 대략적인 적정값을 알기위해 교차검증하면서 무작위로 검색
    # from sklearn.model_selection import RandomizedSearchCV
    # import numpy as np
    # from pprint import pprint
    # # n_estimator :  서브 트리의 갯수를 늘려봄
    # # n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    # n_estimators = [10, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 3000]
    #
    # # max_features : 피쳐 숫자, 또는 루트 피쳐 숫자 둘 중 하나.
    # max_features = ['auto', 'sqrt']
    # ''' max_features 설명 제일 잘 나누기 위해 고려할 피쳐 숫자 설정
    #       int : 각 분할에 최대 피쳐 고려
    #       float : 퍼센트이고 int(max_feature*피쳐 숫자) 피쳐 //잘 모르겠음
    #     “auto”: 최상으로 분할하는데 고려될 피쳐의 숫자로 auto면 최대 숫자는 피쳐 전체 수
    #     “sqrt”: sqrt면 피쳐의 수 루트 값
    #     “log2”: log2면 피쳐의 수 log2 값
    #       None  : auto랑 동일 디폴트 값인듯
    #       분할 검색은 max_features 피쳐보다 더 많이 본다 하더라도 적어도 노드 샘플의 하나의 타당한 분할이 나올때까지 한다.
    #     '''
    # # max_depth : 트리 레벨 최대 수
    # max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]# w적절한 균형이 중요할듯
    # # max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    # max_depth.append(None)
    # # None 이면 노드들은 계속 확장됨 모든 리프노드가 퓨어하거나, 모든 리프노드가 최소 샘플 분할 숫자보다 적게 포함할 때까지. //pure 하다는게 뭘 의미하는지는
    #
    #
    # # 노드를 나누는데 필요한 최소 샘플의 숫자 2, 5, 10
    # min_samples_split = [2, 3, 5, 7, 11, 13, 17, 19] #[2, 5, 10]
    #
    # # 각 리프 노드에 필요한 최소 숫자의 샘플 1, 2, 4
    # min_samples_leaf = [1,2,3,4] #[1, 2, 4]
    #
    # #각 트리를 훈련하는데 선택된 샘플 방법, 부트스트랩 했는지 안했는지.
    # bootstrap = [True, False]
    # # 랜덤한 그리드를 만듦
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    # pprint(random_grid)
    # tuned_random_Forest_randomCV = RandomizedSearchCV(estimator=default_Random_ForestR,
    #                                                   param_distributions=random_grid,
    #                                                   n_iter= 100,  # 100
    #                                                   scoring='neg_mean_squared_error',              #scoring=['neg_mean_absolute_error','neg_mean_squared_error'],
    #                                                   cv=5, verbose=3, random_state=42, n_jobs=-1,
    #                                                   return_train_score=True)
    # # verbose는 숫자 높을수록 많은 정보 n_jobs는 사용 프로세서
    # # cv =5, iter = 100이니까 500번 해서 찾아봄. 여기서 잘 나온 애들을 골라서 그리드 서칭하는 것.
    # # 효과가 좋을수도 안좋을수도 있다
    #
    # # 랜덤 서치 모델에 피팅
    # tuned_random_Forest_randomCV.fit(train_dataset, train_target);
    # print(tuned_random_Forest_randomCV.best_params_) # 최상의 파라미터
    # print("-"*50)
    # print(tuned_random_Forest_randomCV.cv_results_)
    # tuned_random_Forest_randomCV_Regressor = tuned_random_Forest_randomCV.best_estimator_
    # # 최상의 파라미터로 세팅된 RandomForestRegressor()
    # predictions = tuned_random_Forest_randomCV_Regressor.predict(test_dataset)
    # tuned_random_Forest_randomCV_AccuracyWithError = EvaluatingAccuracyWithError(tuned_random_Forest_randomCV_Regressor, predictions,
    #                                                                              train_dataset, train_target,
    #                                                                              test_dataset, test_target, test_target)
    # tuned_random_Forest_randomCV_Regressor_result = {'model': 'randomCV',
    #                                                  'accuracy': tuned_random_Forest_randomCV_AccuracyWithError[0],
    #                                                  'train_error': tuned_random_Forest_randomCV_AccuracyWithError[1],
    #                                                  'validation_error': tuned_random_Forest_randomCV_AccuracyWithError[2],
    #                                                  'n_trees': tuned_random_Forest_randomCV_Regressor.get_params()['n_estimators'],
    #                                                  'n_features': train_dataset.shape[1]}
    # print("무작위 서치를 통한 튜닝")
    # print('향상도 of {:0.2f}%.'.format(100 * ( tuned_random_Forest_randomCV_AccuracyWithError[0]
    #                                           - default_Random_ForestR_AccuracyWithError[0])
    #                                           / default_Random_ForestR_AccuracyWithError[0]))
    # 향상도 = ((뽑은 거 - 기본 정확도)/ 기본 정확도) * 100

    # print("-" * 50)
    print("무작위 서칭 튜닝 결과 (random_Search_RF) - features 98개")
    random_searching_best_estimator_ = {'n_estimators': 3000, 'min_samples_split': 7, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False}
    random_Search_RF = RandomForestRegressor(n_estimators = random_searching_best_estimator_['n_estimators'],
                                             min_samples_split= random_searching_best_estimator_['min_samples_split'],
                                             min_samples_leaf = random_searching_best_estimator_['min_samples_leaf'],
                                             max_features = random_searching_best_estimator_['max_features'],
                                             max_depth = random_searching_best_estimator_['max_depth'],
                                             bootstrap = random_searching_best_estimator_['bootstrap'],
                                             random_state=42)
    random_Search_RF.fit(train_dataset, train_target)
    predictions = random_Search_RF.predict(test_dataset)
    tuned_random_Forest_randomCV_AccuracyWithError = EvaluatingAccuracyWithError(random_Search_RF, predictions, train_dataset, train_target,
                                                   test_dataset, test_target, test_target)
    # visualizationTree("random_cv_tree", random_Search_RF, dataset_list)

    tuned_random_Forest_randomCV_Regressor_result = {'model': 'randomCV',
                                                     'accuracy': tuned_random_Forest_randomCV_AccuracyWithError[0],
                                                     'train_error': tuned_random_Forest_randomCV_AccuracyWithError[1],
                                                     'validation_error': tuned_random_Forest_randomCV_AccuracyWithError[2],
                                                     'n_trees': random_Search_RF.get_params()['n_estimators'],
                                                     'n_features': train_dataset.shape[1]}
    print('향상도 of {:0.2f}%.'.format(100 *
                                    ( tuned_random_Forest_randomCV_AccuracyWithError[0] - default_Random_ForestR_AccuracyWithError[0])
                                    / default_Random_ForestR_AccuracyWithError[0]))
    # #---------------------------------------------------------------------------------------------------------------1.2
    # # 그리드 서치

    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'bootstrap': [True, False],
        'max_depth': [20, 60, 70, 150, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1],
        'min_samples_split': [2, 3, 5, 11],
        'n_estimators': [10, 50, 200, 1000]
        }
    # print("-" * 50, "\n Grid Searching - 전체 15개 변수 사용")
    # default_grid_search = GridSearchCV(estimator=default_Random_ForestR, param_grid=param_grid,
    #                            cv=5, n_jobs=-1, verbose=2, return_train_score=True)
    # print(default_grid_search.get_params().keys())
    # default_grid_search.fit(train_dataset,train_target);
    # #
    # print(default_grid_search.best_params_)
    # print(default_grid_search.cv_results_)
    #
    # default_best_grid = default_grid_search.best_estimator_
    # predictions = default_best_grid.predict(test_dataset)

    #
    # tuned_random_Forest_gridCV_AccuracyWithError = EvaluatingAccuracyWithError(default_grid_search, predictions,
    #                                                                            train_dataset, train_target,
    #                                                                            test_dataset, test_target, test_target)
    # tuned_random_Forest_gridCV_Regressor_result = {'model': 'gridCV',
    #                                                  'accuracy': tuned_random_Forest_gridCV_AccuracyWithError[0],
    #                                                  'train_error': tuned_random_Forest_gridCV_AccuracyWithError[1],
    #                                                  'validation_error': tuned_random_Forest_gridCV_AccuracyWithError[2],
    #                                                  'n_trees': default_best_grid.get_params()['n_estimators'],
    #                                                  'n_features': train_dataset.shape[1]}
    # print("그리드 서칭을 통한 튜닝")
    # print('향상도 of {:0.2f}%.'.format(100 * (tuned_random_Forest_gridCV_AccuracyWithError[0]
    #                                               - default_Random_ForestR_AccuracyWithError[0])
    #                                               / default_Random_ForestR_AccuracyWithError[0]))
    # print("-" * 50)
    #
    print("그리드 서칭 튜닝 결과 (grid_Search_RF) - features 98개")
    default_Random_Forest_grid_searching_best_estimators = {'bootstrap': False, 'max_depth': 60, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400}
    grid_Search_default_RF = RandomForestRegressor(n_estimators = default_Random_Forest_grid_searching_best_estimators['n_estimators'],
                                                   min_samples_split= default_Random_Forest_grid_searching_best_estimators['min_samples_split'],
                                                   min_samples_leaf = default_Random_Forest_grid_searching_best_estimators['min_samples_leaf'],
                                                   max_features = default_Random_Forest_grid_searching_best_estimators['max_features'],
                                                   max_depth = default_Random_Forest_grid_searching_best_estimators['max_depth'],
                                                   bootstrap = default_Random_Forest_grid_searching_best_estimators['bootstrap'],
                                                   random_state=42)
    grid_Search_default_RF.fit(train_dataset, train_target)
    predictions = grid_Search_default_RF.predict(test_dataset)
    tuned_default_grid_AccuracyWithError = EvaluatingAccuracyWithError(grid_Search_default_RF, predictions,
                                                                       train_dataset, train_target, test_dataset,
                                                                       test_target, test_target)
    # # visualizationTree("grid_default_tree", grid_Search_default_RF, dataset_list)
    tuned_default_grid_result = {'model': 'default_grid',
                                 'accuracy': tuned_default_grid_AccuracyWithError[0],
                                 'train_error': tuned_default_grid_AccuracyWithError[1],
                                 'validation_error': tuned_default_grid_AccuracyWithError[2],
                                 'n_trees': grid_Search_default_RF.get_params()['n_estimators'],
                                 'n_features': train_dataset.shape[1]}
    print('향상도 of {:0.2f}%.'.format(100 *
                                    (tuned_default_grid_AccuracyWithError[0] - default_Random_ForestR_AccuracyWithError[0])
                                    / default_Random_ForestR_AccuracyWithError[0]))

    '''
    변수: audience_D11         중요도: 0.44629
    변수: audience_D18         중요도: 0.33236
    변수: audience_D25         중요도: 0.03385
    변수: audience_D14         중요도: 0.03104
    변수: actor_score          중요도: 0.02534
    변수: audience_D13         중요도: 0.01944
    변수: audience_D20         중요도: 0.01684
    변수: audience_D23         중요도: 0.01467
    변수: audience_D15         중요도: 0.01315
    변수: audience_D4          중요도: 0.01114
    변수: show_D12             중요도: 0.00929
    변수: audience_D27         중요도: 0.00443
    변수: audience_D8          중요도: 0.00429
    변수: audience_D9          중요도: 0.00405
    변수: userCount            중요도: 0.00387
    변수: audience_D6          중요도: 0.00237
    변수: audience_D7          중요도: 0.00222
    변수: audience_D19         중요도: 0.00206
    변수: audience_D16         중요도: 0.00175
    변수: show_D11             중요도: 0.00173
    변수: audience_D5          중요도: 0.0017
    변수: audience_D17         중요도: 0.00133
    변수: show_D9              중요도: 0.00091
    변수: audience_D12         중요도: 0.00086
    변수: show_D13             중요도: 0.00084
    변수: show_D8              중요도: 0.00077
    변수: audience_D1          중요도: 0.0007
    변수: audience_D3          중요도: 0.00067
    변수: show_D15             중요도: 0.00066
    변수: audience_D28         중요도: 0.00066
    변수: previous_audience    중요도: 0.00063
    변수: show_D25             중요도: 0.00057
    변수: show_D26             중요도: 0.00056
    변수: audience_D2          중요도: 0.00054
    변수: audience_D26         중요도: 0.00051
    변수: screen_D3            중요도: 0.00045
    변수: screen_D28           중요도: 0.00042
    변수: show_D3              중요도: 0.00039
    변수: director1_score      중요도: 0.00038
    변수: screen_D1            중요도: 0.00033
    변수: audience_D10         중요도: 0.00033
    변수: screen_D25           중요도: 0.00032
    변수: screen_D27           중요도: 0.00028
    변수: show_D19             중요도: 0.00028
    변수: audience_D24         중요도: 0.00028
    변수: audience_D22         중요도: 0.00024
    변수: screen_D11           중요도: 0.00022
    변수: show_D16             중요도: 0.00022
    변수: show_D27             중요도: 0.00022
    변수: show_D18             중요도: 0.0002
    변수: showTm_score         중요도: 0.0002
    변수: show_D1              중요도: 0.00018
    변수: screen_D17           중요도: 0.00016
    변수: show_D17             중요도: 0.00016
    변수: distributor_score    중요도: 0.00016
    변수: previous_screen      중요도: 0.00014
    변수: watchGradeNm_score   중요도: 0.00014
    변수: show_D5              중요도: 0.00013
    변수: screen_D14           중요도: 0.00012
    변수: screen_D20           중요도: 0.00012
    변수: show_D22             중요도: 0.00011
    변수: previous_show        중요도: 0.0001
    변수: starScore            중요도: 0.0001
    변수: screen_D8            중요도: 9e-05
    변수: show_D6              중요도: 9e-05
    변수: show_D21             중요도: 9e-05
    변수: show_D28             중요도: 9e-05
    변수: repNationNm_score    중요도: 9e-05
    변수: screen_D22           중요도: 8e-05
    변수: show_D7              중요도: 8e-05
    변수: show_D10             중요도: 8e-05
    변수: screen_D12           중요도: 7e-05
    변수: screen_D26           중요도: 7e-05
    변수: screen_D19           중요도: 6e-05
    변수: show_D23             중요도: 6e-05
    변수: show_D24             중요도: 6e-05
    변수: screen_D9            중요도: 5e-05
    변수: screen_D16           중요도: 5e-05
    변수: show_D2              중요도: 5e-05
    변수: audience_D21         중요도: 5e-05
    변수: screen_D24           중요도: 4e-05
    변수: show_D14             중요도: 4e-05
    변수: screen_D13           중요도: 3e-05
    변수: screen_D18           중요도: 3e-05
    변수: screen_D21           중요도: 3e-05
    변수: screen_D23           중요도: 3e-05
    변수: screen_D2            중요도: 2e-05
    변수: screen_D4            중요도: 2e-05
    변수: screen_D5            중요도: 2e-05
    변수: screen_D15           중요도: 2e-05
    변수: show_D20             중요도: 2e-05
    변수: prdtYear_score       중요도: 2e-05
    변수: repGenreNm_score     중요도: 2e-05
    변수: screen_D7            중요도: 1e-05
    변수: screen_D10           중요도: 1e-05
    변수: show_D4              중요도: 1e-05
    변수: screen_D6            중요도: 0.0
    -------------------------------------------------- 
    '''


    print("-"*50,"\n Default - 랜덤포레스트에서 찾은 중요 변수 14개") # 95%
    default_Random_ForestR_Importance_features_list = [] #주요변수 담은 리스트공간
    for i in range(0,14):
        default_Random_ForestR_Importance_features_list.append(feature_importances[i][0])
    important_indices = [] # 실제 인덱스 담을 리스트공간
    for i in default_Random_ForestR_Importance_features_list:
        important_indices.append(dataset_list.index(i))

    # visualizationImportance("importance_95", default_Random_ForestR_Importance_features_list, important_indices)

    train_important_dataset = train_dataset.iloc[:, important_indices]
    test_important_dataset = test_dataset.iloc[:, important_indices]

    # 줄인 피쳐로 훈련 시키고, 줄인 피쳐 가진 테스트 셋으로 예측
    default_Random_ForestR_Most_Importance_95 = RandomForestRegressor(random_state=42)
    default_Random_ForestR_Most_Importance_95.fit(train_important_dataset, train_target)
    predictions = default_Random_ForestR_Most_Importance_95.predict(test_important_dataset)
    default_Random_ForestR_Most_Importance_95_AccuracyWithError = EvaluatingAccuracyWithError(default_Random_ForestR_Most_Importance_95,
                                                                            predictions, train_important_dataset,
                                                                            train_target, test_important_dataset,
                                                                            test_target, test_target)
    # visualizationTree("default_MI95_tree", default_Random_ForestR_Most_Importance_95, default_Random_ForestR_Importance_features_list)
    default_Random_ForestR_Most_Importance_95_result = {'model': 'important14',
                                                        'accuracy': default_Random_ForestR_Most_Importance_95_AccuracyWithError[0],
                                                        'train_error': default_Random_ForestR_Most_Importance_95_AccuracyWithError[1],
                                                        'validation_error': default_Random_ForestR_Most_Importance_95_AccuracyWithError[2],
                                                        'n_trees': default_Random_ForestR_Most_Importance_95.get_params()['n_estimators'],
                                                        'n_features': len(default_Random_ForestR_Importance_features_list)}
    print('향상도 of {:0.2f}%.'.format(100 *
                                    (default_Random_ForestR_Most_Importance_95_AccuracyWithError[0] - default_Random_ForestR_AccuracyWithError[0])
                                     / default_Random_ForestR_AccuracyWithError[0]) )
    # print("-" * 50)
    default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators = {'bootstrap': False, 'max_depth': 60, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 3000}
    # # # ---------------------------------------------------------------------------------------------------------------2.1
    print("그리드 서칭 튜닝 결과 (grid_Search_default_RF_95) - features 14개, 중요도 95% 차지")
    grid_Search_default_RF_95 = RandomForestRegressor(
        n_estimators=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['n_estimators'],
        min_samples_split=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['min_samples_split'],
        min_samples_leaf=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['min_samples_leaf'],
        max_features=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['max_features'],
        max_depth=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['max_depth'],
        bootstrap=default_Random_ForestR_Most_Importance_95_grid_searching_best_estimators['bootstrap'],
        random_state=42)

    grid_Search_default_RF_95.fit(train_dataset, train_target)
    predictions = grid_Search_default_RF_95.predict(test_dataset)
    grid_Search_default_RF_95_AccuracyWithError = EvaluatingAccuracyWithError(grid_Search_default_RF_95, predictions,
                                                                              train_important_dataset,
                                                                              train_target, test_important_dataset, test_target,
                                                                              test_target)
    # # visualizationTree("grid_MI99_tree", grid_Search_default_RF_99, default_Random_ForestR_Importance_features_list)
    tuned_MI95_grid_result = {'model': 'grid_95',
                              'accuracy': grid_Search_default_RF_95_AccuracyWithError[0],
                              'train_error': grid_Search_default_RF_95_AccuracyWithError[1],
                              'validation_error': grid_Search_default_RF_95_AccuracyWithError[2],
                              'n_trees': grid_Search_default_RF_95.get_params()['n_estimators'],
                              'n_features': len(default_Random_ForestR_Importance_features_list)}

    print('향상도 of {:0.2f}%.'.format(100 *
                                    (grid_Search_default_RF_95_AccuracyWithError[0] -
                                     default_Random_ForestR_AccuracyWithError[0])
                                    / default_Random_ForestR_AccuracyWithError[0]))

    # visualizationTree("default_tree_MI_RF", default_Random_ForestR_Most_Importance_RF,
    #                   default_Random_ForestR_Importance_features_list)

    # print("-" * 50)
    # # ---------------------------------------------------------------------------------------------------------------2.5
    # print("-" * 50, "\n Grid Searching - 랜덤포레스트에서 찾은 중요 변수 6개")
    # # Instantiate the grid search model
    # grid_Search_default_RF_95 = GridSearchCV(estimator=default_Random_ForestR_Most_Importance_95, param_grid=param_grid,
    #                                  cv=5, n_jobs=3, verbose=2, return_train_score=True)
    # grid_Search_default_RF_95.fit(train_important_dataset, train_target);
    # #
    # print(grid_Search_default_RF_95.best_params_)
    # print(grid_Search_default_RF_95.cv_results_)
    # best_grid_MI_RF = grid_Search_default_RF_95.best_estimator_
    # predictions = grid_Search_default_RF_95.predict(test_important_dataset)
    # grid_Search_default_RF_95_AccuracyWithError = EvaluatingAccuracyWithError(default_Random_ForestR_Most_Importance_95,
    #                                                                           predictions, train_important_dataset,
    #                                                                           train_target, test_important_dataset,
    #                                                                           test_target, test_target)
    # print('향상도 of {:0.2f}%.'.format(100 *
    #                                 (grid_Search_default_RF_95_AccuracyWithError[0]
    #                                  - default_Random_ForestR_AccuracyWithError[0])
    #                                 / default_Random_ForestR_AccuracyWithError[0]) )
    # # #얘는 이거
    # #---------------------------------------------------------------------------------------------------------------2.2

    print("-" * 50, "\n Default - 랜덤포레스트에서 찾은 중요 변수 42개")  # 99%
    default_Random_ForestR_Importance_features_list = []  # 주요변수 담은 리스트공간
    for i in range(0, 42):
        default_Random_ForestR_Importance_features_list.append(feature_importances[i][0])
    important_indices = []  # 실제 인덱스 담을 리스트공간
    for i in default_Random_ForestR_Importance_features_list:
        important_indices.append(dataset_list.index(i))
    # visualizationImportance("importance_99", default_Random_ForestR_Importance_features_list, important_indices)
    train_important_dataset = train_dataset.iloc[:, important_indices]
    test_important_dataset = test_dataset.iloc[:, important_indices]
    all_important_dataset = dataset.iloc[:, important_indices]
    # # 줄인 피쳐로 훈련 시키고, 줄인 피쳐 가진 테스트 셋으로 예측
    default_Random_ForestR_Most_Importance_99 = RandomForestRegressor(random_state=42)
    default_Random_ForestR_Most_Importance_99.fit(train_important_dataset, train_target)
    predictions = default_Random_ForestR_Most_Importance_99.predict(test_important_dataset)
    default_Random_ForestR_Most_Importance_99_AccuracyWithError = EvaluatingAccuracyWithError(
        default_Random_ForestR_Most_Importance_99,
        predictions, train_important_dataset,
        train_target, test_important_dataset,
        test_target, test_target)
    # visualizationTree("default_MI95_tree", default_Random_ForestR_Most_Importance_95, default_Random_ForestR_Importance_features_list)
    default_Random_ForestR_Most_Importance_99_result = {'model': 'important42', 'accuracy': default_Random_ForestR_Most_Importance_99_AccuracyWithError[0],
                                                        'train_error':default_Random_ForestR_Most_Importance_99_AccuracyWithError[1],
                                                        'validation_error':default_Random_ForestR_Most_Importance_99_AccuracyWithError[2],
                                                        'n_trees': default_Random_ForestR_Most_Importance_99.get_params()['n_estimators'],
                                                        'n_features': len(default_Random_ForestR_Importance_features_list)}
    print('향상도 of {:0.2f}%.'.format(100 *
                                    (default_Random_ForestR_Most_Importance_99_AccuracyWithError[0] -
                                     default_Random_ForestR_AccuracyWithError[0])
                                    / default_Random_ForestR_AccuracyWithError[0]))

    default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators = {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 3000}
    print("그리드 서칭 튜닝 결과 (grid_Search_default_RF_99) - features 42개, 중요도 99% 차지")
    grid_Search_default_RF_99 = RandomForestRegressor(
        n_estimators=default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators['n_estimators'],
        min_samples_split=default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators['min_samples_split'],
        min_samples_leaf=default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators['min_samples_leaf'],
        max_features=default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators['max_features'],
        max_depth=default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators['max_depth'],
        bootstrap=default_Random_ForestR_Most_Importance_99_grid_searching_best_estimators['bootstrap'],
        random_state=42)

    grid_Search_default_RF_99.fit(train_dataset, train_target)
    predictions = grid_Search_default_RF_99.predict(test_dataset)
    grid_Search_default_RF_99_AccuracyWithError = EvaluatingAccuracyWithError(grid_Search_default_RF_99, predictions,
                                                            train_important_dataset,
                                                            train_target, test_important_dataset, test_target,
                                                            test_target)
    # visualizationTree("grid_MI99_tree", grid_Search_default_RF_99, default_Random_ForestR_Importance_features_list)
    tuned_MI99_grid_result = {'model': 'grid_99',
                              'accuracy': grid_Search_default_RF_99_AccuracyWithError[0],
                              'train_error': grid_Search_default_RF_99_AccuracyWithError[1],
                              'validation_error': grid_Search_default_RF_99_AccuracyWithError[2],
                              'n_trees': grid_Search_default_RF_99.get_params()['n_estimators'],
                              'n_features': len(default_Random_ForestR_Importance_features_list)}

    print('향상도 of {:0.2f}%.'.format(100 *
                                    (grid_Search_default_RF_99_AccuracyWithError[0] -
                                     default_Random_ForestR_AccuracyWithError[0])
                                    / default_Random_ForestR_AccuracyWithError[0]))
    #
    # # visualizationTree("default_tree_MI_RF", default_Random_ForestR_Most_Importance_RF,
    # #                   default_Random_ForestR_Importance_features_list)
    # print("-" * 50)
    # # # ---------------------------------------------------------------------------------------------------------------2.5
    # # ---------------------------------------------------------------------------------------------------------------2.1
    # print("-" * 50, "\n Grid Searching - 랜덤포레스트에서 찾은 중요 변수 13개")
    # # Instantiate the grid search model
    # grid_Search_default_RF_99 = GridSearchCV(estimator=default_Random_ForestR_Most_Importance_99, param_grid=param_grid,
    #                                          cv=5, n_jobs=3, verbose=2, return_train_score=True)
    # grid_Search_default_RF_99.fit(train_important_dataset, train_target);
    # #
    # print(grid_Search_default_RF_99.best_params_)
    # print(grid_Search_default_RF_99.cv_results_)
    # best_grid_MI_RF = grid_Search_default_RF_99.best_estimator_
    # predictions = grid_Search_default_RF_99.predict(test_important_dataset)
    # grid_Search_default_RF_99_AccuracyWithError = EvaluatingAccuracyWithError(default_Random_ForestR_Most_Importance_99,
    #                                                                           predictions, train_important_dataset,
    #                                                                           train_target, test_important_dataset,
    #                                                                           test_target, test_target)
    # print('향상도 of {:0.2f}%.'.format(100 *
    #                                 (grid_Search_default_RF_99_AccuracyWithError[0]
    #                                  - default_Random_ForestR_AccuracyWithError[0])
    #                                 / default_Random_ForestR_AccuracyWithError[0]))
    # # 얘는 이거
    # # ---------------------------------------------------------------------------------------------------------------2.2
    #
    # 튜닝 결과 비교하기

    list = [tuned_random_Forest_randomCV_Regressor_result, tuned_default_grid_result,
            default_Random_ForestR_Most_Importance_95_result, tuned_MI95_grid_result,
            default_Random_ForestR_Most_Importance_99_result, tuned_MI99_grid_result]
    ## Comparison
    comparison = {'model': [default_Random_ForestR_result['model']],
                  'accuracy':[round(default_Random_ForestR_result['accuracy'], 2)],
                  'train_error':  [round(default_Random_ForestR_result['train_error'], 2)],
                  'validation_error': [round(default_Random_ForestR_result['validation_error'], 2)],
                  'n_features': [default_Random_ForestR_result['n_features']],
                  'n_trees':  [int(default_Random_ForestR_result['n_trees'])],
                  'differenceTEtoVE':[round(abs(default_Random_ForestR_result['validation_error']
                                                - default_Random_ForestR_result['train_error']), 2)]}
    print(comparison)
    for model in list:
        comparison['model'].append(model['model'])
        comparison['accuracy'].append(round(model['accuracy'], 2))
        comparison['train_error'].append(round(model['train_error'], 2))
        comparison['validation_error'].append(round(model['validation_error'], 2))
        comparison['n_features'].append(model['n_features'])
        comparison['n_trees'].append(int(model['n_trees']))
        comparison['differenceTEtoVE'].append(round(abs(model['validation_error']- model['train_error']), 2))

    comparison = pd.DataFrame.from_dict(comparison, orient='columns')
    print(comparison[['model', 'accuracy', 'train_error','validation_error', 'n_features', 'n_trees']])
    # # comparison.to_csv(os.getcwd()+"/../data/99_output_RandomForest_D28_Comparison.csv", encoding="utf-8")
    # #
    # ## Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    #%matplotlib inline

    plt.style.use('fivethirtyeight')

    # Model Comparison Plot
    xvalues = range(0,len(comparison))
    plt.subplots(1, 3, figsize=(16, 9))
    plt.subplot(131)
    plt.bar(xvalues, comparison['accuracy'], color='b', edgecolor='k', linewidth=1.8)
    plt.xticks(xvalues, comparison['model'], rotation=45, fontsize=7)
    plt.ylim(ymin=85, ymax=95)
    plt.xlabel('model');
    plt.ylabel('Accuracy (%)');
    plt.title('Accuracy Comparison', y = 1.05);

    plt.subplot(132)
    plt.bar(xvalues, comparison['train_error'], color='g', edgecolor='k', linewidth=1)
    plt.xticks(xvalues, comparison['model'], rotation=45, fontsize=7)
    plt.xlabel('model');
    plt.ylabel('Train Error');
    plt.title('Train Error Comparison',y = 1.05);

    plt.subplot(133)
    plt.bar(xvalues, comparison['validation_error'], color='pink', edgecolor='k', linewidth=1)
    plt.xticks(xvalues, comparison['model'], rotation=45, fontsize=7)
    # plt.ylim(ymin=max(comparison['validation_error']) - min(comparison['validation_error']) / 3,
    #          ymax=max(comparison['validation_error']) + min(comparison['validation_error']) / 3)
    plt.xlabel('model');
    plt.ylabel('Validation Error (deg)');
    plt.title('Validation Error Comparison',y = 1.05);
    plt.show();

    print("-"*50)
    # choice grid_99
    print("grid_search_99 resultAll")
    resultAllPredict(grid_Search_default_RF_99, all_important_dataset,"RandomForest_D28")
    #
    # # resultAllPredict(grid_Search_default_RF_99, all_important_dataset)