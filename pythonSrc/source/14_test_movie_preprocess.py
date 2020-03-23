# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import copy


# data_set = pd.read_csv(os.getcwd() + '/../data/z_testData.csv',index_col =0)
# data_set = data_set.rename(columns={"director1":"director"})
#
# # data_set = data_set.drop(["director2"],axis = 1)
# print(data_set.columns)
# print(data_set.shape)
# print()
# def month_change(i):#str(data_set["openDt"][0])[4:6]
#     if i == "01":
#         return 'jan'
#     elif i == '02':
#         return 'feb'
#     elif i == '03':
#         return 'mar'
#     elif i == '04':
#         return 'apr'
#     elif i == '05':
#         return 'may'
#     elif i == '06':
#         return 'jun'
#     elif i == '07':
#         return 'jul'
#     elif i == '08':
#         return 'aug'
#     elif i == '09':
#         return 'sep'
#     elif i == '10':
#         return 'oct'
#     elif i == '11':
#         return 'nov'
#     else:
#         return 'dec'
# # 아 이게 이상하게 나왔네. 중간에 월만 남겨둔게 아니니까 이렇게 나왔지
# # 겨우 바꿨네
# for i in range(0,len(data_set["openDt"])):
#     data_set.loc[i,"openDt"] = str(data_set["openDt"][i])[4:6]
# data_set["openDt"] = data_set["openDt"].apply(month_change)
# # print(data_set["openDt"])
#
# def grade_change(i):
#     if i == '전체관람가':
#         return 'ALL'
#     elif i == '12세이상관람가':
#         return 'olderThan12'
#     elif i == '15세관람가' or i == '15세이상관람가':
#         return 'olderThan15'
#     elif i == '청소년관람불가':
#         return 'olderThan19'
#     else:
#         return i
# data_set["watchGradeNm"] = data_set["watchGradeNm"].apply(grade_change)
# # print(data_set["watchGradeNm"])
# data_set["showTm"] = data_set["showTm"].fillna(0)
# data_set["showTm"] = data_set["showTm"].astype(int)
#
# def time_change(i):
#     if i < 90:
#         return 'under_90'
#     elif i >= 90 and i < 120:
#         return '90_120'
#     elif i >=120 and i < 150:
#         return '120_150'
#     else:
#         return '150_up'
#
# data_set["showTm"] = data_set["showTm"].apply(time_change)
# #print(data_set["showTm"])
#
# def nation_change(i):
#     if i == '중국' or i == '홍콩' or i == '대만':
#         return 'china'
#     elif i == '아이슬란드' or i == '우크라이나' or i == '체코' or i == '노르웨이' or\
#             i == ' 오스트리아' or i == '덴마크' or i == '러시아' or i == '이탈리아' or \
#             i == '벨기에' or i == '폴란드' or i == '스페인' or i == '핀란드' or \
#             i == '스웨덴' or i == '스위스' or i == '영국' or i == '프랑스' or i == '헝가리' \
#             or i == '독일' or i == '아일랜드' or i == '그리스' or i == '터키':
#         return 'europe'
#     elif i == '태국' or i == '말레이시아' or i == '호주' or i == '인도네시아':
#         return 'other_asia'
#     elif i == '캐나다' or i == '페루' or i == '멕시코' or i == '뉴질랜드' or i == '아르헨티나' \
#             or i == '이란' or i == '남아프리카공화국' or i == '칠레' or i == '브라질' or i == '이스라엘':
#         return 'other_nation'
#     elif i == '한국':
#         return 'korea'
#     elif i == '인도':
#         return 'India'
#     elif i == '미국':
#         return 'america'
#     elif i == '일본':
#         return 'japan'
#
# data_set["repNationNm"] = data_set["repNationNm"].apply(nation_change)
# # #print(data_set["repNationNm"])
# #
# #
# def genre_change(i):
#     if i == '드라마' or i == '멜로/로맨스':
#         return 'drama_romance'
#     elif i == '전쟁' or i == '액션':
#         return 'war_action'
#     elif i == '공포(호러)' or i == '미스터리':
#         return 'horror_mystery'
#     elif i == '범죄' or i == '스릴러':
#         return 'crime_thriller'
#     elif i == 'SF':
#         return 'SF'
#     elif i == '판타지' or i == '어드벤처':
#         return 'fantasy_adventure'
#     elif i == '애니메이션' or i == '가족':
#         return 'family_animation'
#     elif i == '코미디':
#         return 'comedy'
#     elif i == '다큐멘터리':
#         return 'documentary'
#     elif i == '공연' or i == '뮤지컬':
#         return 'performance_musical'
#     elif i == '사극' or i == '서부극(웨스턴)':
#         return 'historical'
#     else:
#         return i
#
# data_set["repGenreNm"] = data_set["repGenreNm"].apply(genre_change)
# # #print(data_set["repGenreNm"])
# #
# def company_change(i):
#     if '디즈니' in i:
#         return 'walt_disney'
#     elif '폭스' in i:
#         return 'twentieth_century_fox'
#     elif '쇼박스' in i:
#         return 'showbox'
#     elif '브러더스' in i:
#         return 'warnerbros'
#     elif '메가박스'in i:
#         return 'megabox'
#     elif '유니버' in i:
#         return 'universal'
#     elif '넥스트' in i or 'NEW' in i or 'new' in i:
#         return 'next'
#     elif '와우' in i:
#         return 'wowpictures'
#     elif '롯데' in i:
#         return 'lotte'
#     elif '오퍼스' in i:
#         return 'opus'
#     elif '리틀' in i:
#         return 'littlebig'
#     elif '씨네그루' in i:
#         return 'cineguru'
#     elif '소니' in i:
#         return 'sony'
#     elif '판씨네마' in i:
#         return 'pancine'
#     elif '이수' in i:
#         return 'isu'
#     elif '씨제이' in i or 'CJ' in i or 'cj' in i or '씨지브이' in i or 'CGV' in i:
#         return 'cjenm'
#     else:
#         return 'other_company'
# data_set["distributor"] = data_set["distributor"].apply(company_change)
# # data_set["producedCompany"] = data_set["producedCompany"].apply(company_change)
#
# #print(data_set["distributor"])
# #print(data_set["producedCompany"])
#
# cate_set = copy.deepcopy(data_set[['movieCd','director', 'openDt', 'prdtYear', 'repNationNm',
#                                    'repGenreNm', 'showTm', 'watchGradeNm', 'actor1', 'actor2','actor3','actor4',
#                                    'actor5','actor6', 'distributor', 'starScore','userCount','final_audience']])
# #
# def resizeValue(i):
#     if i < 500000:
#         return 2
#     elif i >= 500000 and i < 1000000:
#         return 4
#     elif i >= 1000000 and i < 2000000:
#         return 6
#     elif i >= 2000000 and i < 3000000:
#         return 8
#     elif i >= 3000000 and i < 4000000:
#         return 10
#     elif i >= 4000000 and i < 5000000:
#         return 12
#     elif i >= 5000000 and i < 6000000:
#         return 14
#     elif i >= 6000000 and i < 7000000:
#         return 16
#     elif i >= 7000000 and i < 8000000:
#         return 18
#     elif i >= 8000000 and i < 9000000:
#         return 20
#     elif i >= 9000000 and i < 10000000:
#         return 22
#     else:
#         return 24
# # def score_change(i):
# #     if i < 1000000:
# #         return 1
# #     elif i < 2000000 and i >= 1000000:
# #         return 2
# #     elif i < 3000000 and i >= 2000000:
# #         return 3
# #     elif i < 4000000 and i >= 3000000:
# #         return 4
# #     elif i < 5000000 and i >= 4000000:
# #         return 5
# #     elif i < 6000000 and i >= 5000000:
# #         return 6
# #     elif i < 7000000 and i >= 6000000:
# #         return 7
# #     elif i < 8000000 and i >= 7000000:
# #         return 8
# #     elif i < 9000000 and i >= 8000000:
# #         return 9
# #     else:
# #         return 10
# #
# cate_set["final_audience"] = cate_set["final_audience"].apply(resizeValue)
# # 기존 개봉날짜, 제작연도, 국가, 장르, 쇼타임, 등급 얘네는 점수를 가져다가 그대로 쓰고
# # 배우, 감독, 배급사 그대로쓰는데 없는애들은 1점으로 매겨서
# cate_set.to_csv(os.getcwd() + "/../data/z_test_preprocessing_MovieData.csv",encoding="utf-8")

test_set = pd.read_csv(os.getcwd() + '/../data/z_test_preprocessing_MovieData.csv',index_col =0)

openDt_score_set = pd.read_csv(os.getcwd() + '/../score/openDt_score.csv',index_col =0)          # 개봉날짜
prdtYear_score_set = pd.read_csv(os.getcwd() + '/../score/prdtYear_score.csv',index_col =0)      # 제작연도
nation_score_set = pd.read_csv(os.getcwd() + '/../score/nation_score.csv',index_col =0)          # 국가
genre_score_set = pd.read_csv(os.getcwd() + '/../score/genre_score.csv',index_col =0)            # 장르
showTm_score_set = pd.read_csv(os.getcwd() + '/../score/showTm_score.csv',index_col =0)          # 쇼타임
watchGrade_score_set = pd.read_csv(os.getcwd() + '/../score/watchGrade_score.csv',index_col =0)  # 등급
director_score_set = pd.read_csv(os.getcwd() + '/../score/director_score.csv',index_col =0)      # 감독
actor_score_set = pd.read_csv(os.getcwd() + '/../score/actor_score.csv',index_col =0)            # 배우
distributor_score_set = pd.read_csv(os.getcwd() + '/../score/distributor_score.csv',index_col =0)# 배급사

# openDt_score_set      = openDt_score_set.rename(columns={"final_audience":"openDt_score"})
# prdtYear_score_set    = prdtYear_score_set.rename(columns={"final_audience":"prdtYear_score"})
# nation_score_set      = nation_score_set.rename(columns={"final_audience":"repNationNm_score"})
# genre_score_set       = genre_score_set.rename(columns={"final_audience":"repGenreNm_score"})
# showTm_score_set      = showTm_score_set.rename(columns={"final_audience":"showTm_score"})
# watchGrade_score_set  = watchGrade_score_set.rename(columns={"final_audience":"watchGradeNm_score"})
# director_score_set    = director_score_set.rename(columns={"final_audience":"director_score"})
actor_score_set       = actor_score_set.rename(columns={"final_audience_y":"actor_score"})
# distributor_score_set = distributor_score_set.rename(columns={"final_audience":"distributor_score"})
# #
# print(openDt_score_set)
# print(prdtYear_score_set)
# print(nation_score_set)
# print(genre_score_set)
# print(showTm_score_set)
# print(watchGrade_score_set)
# print(director_score_set)
# print(actor_score_set)
# print(distributor_score_set)
# print(test_set)

# #
#
test_set= test_set.merge(openDt_score_set,how='left', on = "openDt")
test_set= test_set.merge(prdtYear_score_set, how='left',on = "prdtYear")
test_set= test_set.merge(nation_score_set, how='left',on = "repNationNm")
test_set= test_set.merge(genre_score_set, how='left',on = "repGenreNm")
test_set= test_set.merge(showTm_score_set, how='left',on = "showTm")
test_set= test_set.merge(watchGrade_score_set, how='left',on = "watchGradeNm")
test_set= test_set.merge(director_score_set, how='left',on = "director")
test_set=test_set.fillna(1)
test_set= test_set.merge(distributor_score_set, how='left',on = "distributor")
test_set=test_set.fillna(1)
#
# print(actor_score_set["actor_score"][0])
for i in range(1,7):
    actors = "actor" + str(i)
    actor_scores = "actor_score" + str(i)
    test_set = pd.merge(test_set, actor_score_set, how='left', left_on = actors, right_on= "actor")
    test_set = test_set.fillna(actor_score_set["actor_score"][0])
    test_set = test_set.drop("actor",axis=1)
    test_set = test_set.rename(columns={"actor_score":actor_scores})

# test_set.to_csv("tes2.csv")
#
test_set["actor_score"] = test_set["actor_score1"]+ test_set["actor_score2"] + test_set["actor_score3"]+ test_set["actor_score4"]+ test_set["actor_score5"] + test_set["actor_score6"]
#
# #
# # # actor = actor[["movieCd","actor_score"]]
# # #
# # # actor = actor.fillna(0)
# # # #actor.to_csv("actortest.csv",encoding="utf-8")
# # # #director	openDt	prdtYear	repNationNm	repGenreNm	showTm	watchGradeNm	actor1	actor2	actor3	actor4	actor5	actor6	distributor	final_audience
test_value_data_set = test_set.drop(['director', 'openDt', 'prdtYear', 'repNationNm', 'repGenreNm', 'showTm',
                                     'watchGradeNm', 'actor1', 'actor2', 'actor3','actor4','actor5','actor6',
                                     'actor_score1', 'actor_score2', 'actor_score3','actor_score4', 'actor_score5',
                                     'actor_score6', 'distributor','final_audience'],axis=1)
# #movieCd	director_score	openDt_score	prdtYear_score	repNationNm_score	repGenreNm_score	showTm_score	watchGradeNm_score	distributor_score	actor_score	starScore	userCount
test_value_data_set["director_score"] = round(test_value_data_set["director_score"],1)
test_value_data_set["openDt_score"] = round(test_value_data_set["openDt_score"],1)
test_value_data_set["prdtYear_score"] = round(test_value_data_set["prdtYear_score"],1)
test_value_data_set["repNationNm_score"] = round(test_value_data_set["repNationNm_score"],1)
test_value_data_set["repGenreNm_score"] = round(test_value_data_set["repGenreNm_score"],1)
test_value_data_set["showTm_score"] = round(test_value_data_set["showTm_score"],1)
test_value_data_set["watchGradeNm_score"] = round(test_value_data_set["watchGradeNm_score"],1)
test_value_data_set["distributor_score"] = round(test_value_data_set["distributor_score"],1)
test_value_data_set["actor_score"] = round(test_value_data_set["actor_score"],1)
# test_value_data_set.to_csv("testt.csv")
test_value_data_set = pd.DataFrame(test_value_data_set, columns=['movieCd','director_score','openDt_score','prdtYear_score',
                                                                 'repNationNm_score','repGenreNm_score','showTm_score',
                                                                 'watchGradeNm_score','distributor_score',
                                                                 'actor_score','starScore','userCount'])
value_set = pd.read_csv(os.getcwd()+"/../data/z_testData.csv",index_col=0)
print(value_set.columns)
# # 정아가 hi라고 이름 지으랬음
value_set = value_set[["movieCd", 'previous_screen','screen_D1', 'screen_D2', 'screen_D3','screen_D4', 'screen_D5', 'screen_D6',
                       'screen_D7', 'screen_D8', 'screen_D9', 'screen_D10', 'screen_D11', 'screen_D12', 'screen_D13',
                       'screen_D14', 'screen_D15', 'screen_D16', 'screen_D17','screen_D18', 'screen_D19', 'screen_D20',
                       'screen_D21', 'screen_D22', 'screen_D23', 'screen_D24', 'screen_D25', 'screen_D26', 'screen_D27',
                       'screen_D28', 'previous_show', 'show_D1', 'show_D2', 'show_D3','show_D4', 'show_D5', 'show_D6',
                       'show_D7', 'show_D8', 'show_D9', 'show_D10', 'show_D11', 'show_D12', 'show_D13', 'show_D14',
                       'show_D15', 'show_D16', 'show_D17', 'show_D18', 'show_D19', 'show_D20', 'show_D21', 'show_D22',
                       'show_D23', 'show_D24', 'show_D25', 'show_D26', 'show_D27', 'show_D28', 'previous_audience',
                       'audience_D1', 'audience_D2','audience_D3', 'audience_D4', 'audience_D5', 'audience_D6',
                       'audience_D7', 'audience_D8', 'audience_D9', 'audience_D10', 'audience_D11', 'audience_D12',
                       'audience_D13', 'audience_D14', 'audience_D15', 'audience_D16', 'audience_D17', 'audience_D18',
                       'audience_D19', 'audience_D20', 'audience_D21', 'audience_D22', 'audience_D23', 'audience_D24',
                       'audience_D25', 'audience_D26', 'audience_D27', 'audience_D28', 'final_audience']]

test_value_data_set = test_value_data_set.merge(value_set,on="movieCd")
test_value_data_set.to_csv(os.getcwd()+"/../checkcheck/z_test_data.csv",encoding="utf-8")
# # # '''
# # 스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을 적용하여
# # 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
# # 스케일링은 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지하고
# # 독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜
# # 최적화 과정에서의 안정성 및 수렴 속도를 향상시킨다.
# #  scale(X): 기본 스케일. 평균과 표준편차 사용
# #  robust_scale(X): 중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
# # '''
#
# # from sklearn.preprocessing import StandardScaler
# #
# # movieCd = test_value_data_set["movieCd"]
# # dd_set = test_value_data_set.drop(["movieCd"],axis=1)
# # dataset = dd_set.ix[:,:-1]
# # target = dd_set.ix[:,-1]
# #
# # scaler = StandardScaler(with_mean=False)
# # dataset_scaled = scaler.fit_transform(dataset) # = fit / transform
# #
# # dataset_forDF= pd.DataFrame(dataset_scaled,columns=dataset.columns) # 스케일링된 데이터셋
# # target_forDF= pd.DataFrame(dd_set.ix[:,-1],columns=["final_audience"]) # 스케일링 안된 그냥 타겟
# #
# # scaled_DF = pd.concat([movieCd, dataset_forDF, target_forDF], axis=1)
# # scaled_DF.to_csv(os.getcwd()+"/../data/test_value_data.csv",encoding="utf-8")
#
#
# # 애트리뷰트 순서 바꿔서 저장함 openDt 값도 정상적으로 나옴
# # string으로 되어있던 star Score, user Count는 empty(" ")값 때문이었음. (null, nan , None 아님) 비어있던 공간 없애니
# # 자동으로 타입 변경됨
# # 5만명 이상으로 추림 // 1~5만 사이는 관객수가 너무 적어서 예측 변동폭이 너무 큼.
# # -> 얼마 차이 안나지만 예측률을 크게 떨어뜨리는 요소