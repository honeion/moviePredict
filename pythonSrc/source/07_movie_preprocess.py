# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import copy

data_set = pd.read_csv(os.getcwd() + '/../data/08_completed_MovieData.csv',index_col =0)
data_set = data_set.rename(columns={"director1":"director"})
data_set = data_set.drop(["director2"],axis = 1)
data_set = data_set[data_set["final_audience"]>50000].reset_index(drop=True)
print(data_set.columns)
print(data_set.shape)
print()
def month_change(i):
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
##str(data_set["openDt"][0])[4:6]
# # data_set = pd.read_csv(os.getcwd() + '/sfinalMovieData.csv',index_col =0)
# # star_data_set = pd.read_csv(os.getcwd() + '/movie_data_naverCrawl.csv',index_col =0)
# # data_set["starScore"] = star_data_set["starScore"]
# # data_set["userCount"] = star_data_set["userCount"]
# # data_set["openDt"] = data_set["openDt"].apply(lambda  x:str(x)[4:6])
# # 감독1만 남기고 감독2는 제거, 감독1은 감독으로 이름 변경
# # data_set = pd.read_csv(os.getcwd() + '/../data/08_completed_MovieData.csv',index_col =0)
for i in range(0,len(data_set["openDt"])):
    data_set.loc[i,"openDt"] = str(data_set["openDt"][i])[4:6]
data_set["openDt"] = data_set["openDt"].apply(month_change)
# print(data_set["openDt"])

def grade_change(i):
    if i == '전체관람가':
        return 'ALL'
    elif i == '12세이상관람가':
        return 'olderThan12'
    elif i == '15세관람가' or i == '15세이상관람가':
        return 'olderThan15'
    elif i == '청소년관람불가':
        return 'olderThan19'
    else:
        return i
data_set["watchGradeNm"] = data_set["watchGradeNm"].apply(grade_change)
data_set["showTm"] = data_set["showTm"].fillna(0)
data_set["showTm"] = data_set["showTm"].astype(int)

def time_change(i):
    if i < 90:
        return 'under_90'
    elif i >= 90 and i < 120:
        return '90_120'
    elif i >=120 and i < 150:
        return '120_150'
    else:
        return '150_up'

data_set["showTm"] = data_set["showTm"].apply(time_change)


def nation_change(i):
    if i == '중국' or i == '홍콩' or i == '대만':
        return 'china'
    elif i == '아이슬란드' or i == '우크라이나' or i == '체코' or i == '노르웨이' or\
            i == ' 오스트리아' or i == '덴마크' or i == '러시아' or i == '이탈리아' or \
            i == '벨기에' or i == '폴란드' or i == '스페인' or i == '핀란드' or \
            i == '스웨덴' or i == '스위스' or i == '영국' or i == '프랑스' or i == '헝가리' \
            or i == '독일' or i == '아일랜드' or i == '그리스':
        return 'europe'
    elif i == '태국' or i == '말레이시아' or i == '호주' or i == '인도네시아':
        return 'other_asia'
    elif i == '캐나다' or i == '페루' or i == '멕시코' or i == '뉴질랜드' or i == '아르헨티나' \
            or i == '이란' or i == '남아프리카공화국' or i == '칠레' or i == '브라질' or i == '이스라엘':
        return 'other_nation'
    elif i == '한국':
        return 'korea'
    elif i == '인도':
        return 'India'
    elif i == '미국':
        return 'america'
    elif i == '일본':
        return 'japan'


data_set["repNationNm"] = data_set["repNationNm"].apply(nation_change)

def genre_change(i):
    if i == '드라마' or i == '멜로/로맨스':
        return 'drama_romance'
    elif i == '전쟁' or i == '액션':
        return 'war_action'
    elif i == '공포(호러)' or i == '미스터리':
        return 'horror_mystery'
    elif i == '범죄' or i == '스릴러':
        return 'crime_thriller'
    elif i == 'SF':
        return 'SF'
    elif i == '판타지' or i == '어드벤처':
        return 'fantasy_adventure'
    elif i == '애니메이션' or i == '가족':
        return 'family_animation'
    elif i == '코미디':
        return 'comedy'
    elif i == '다큐멘터리':
        return 'documentary'
    elif i == '공연' or i == '뮤지컬':
        return 'performance_musical'
    elif i == '사극' or i == '서부극(웨스턴)':
        return 'historical'
    else:
        return i

data_set["repGenreNm"] = data_set["repGenreNm"].apply(genre_change)
# #print(data_set["repGenreNm"])
#
def company_change(i):
    if '디즈니' in i:
        return 'walt_disney'
    elif '폭스' in i:
        return 'twentieth_century_fox'
    elif '쇼박스' in i:
        return 'showbox'
    elif '브러더스' in i:
        return 'warnerbros'
    elif '메가박스'in i:
        return 'megabox'
    elif '유니버' in i:
        return 'universal'
    elif '넥스트' in i or 'NEW' in i or 'new' in i:
        return 'next'
    elif '와우' in i:
        return 'wowpictures'
    elif '롯데' in i:
        return 'lotte'
    elif '오퍼스' in i:
        return 'opus'
    elif '리틀' in i:
        return 'littlebig'
    elif '씨네그루' in i:
        return 'cineguru'
    elif '소니' in i:
        return 'sony'
    elif '판씨네마' in i:
        return 'pancine'
    elif '이수' in i:
        return 'isu'
    elif '씨제이' in i or 'CJ' in i or 'cj' in i or '씨지브이' in i or 'CGV' in i:
        return 'cjenm'
    else:
        return 'other_company'
data_set["distributor"] = data_set["distributor"].apply(company_change)
# data_set["producedCompany"] = data_set["producedCompany"].apply(company_change)

#print(data_set["distributor"])
#print(data_set["producedCompany"])

cate_set = copy.deepcopy(data_set[['movieCd', 'starScore','userCount','director', 'openDt', 'prdtYear', 'repNationNm',
                                   'repGenreNm', 'showTm', 'watchGradeNm', 'actor1', 'actor2','actor3','actor4',
                                   'actor5','actor6', 'distributor','final_audience']])
#
def resizeValue(i):
    if i < 500000:
        return 2
    elif i >= 500000 and i < 1000000:
        return 4
    elif i >= 1000000 and i < 2000000:
        return 6
    elif i >= 2000000 and i < 3000000:
        return 8
    elif i >= 3000000 and i < 4000000:
        return 10
    elif i >= 4000000 and i < 5000000:
        return 12
    elif i >= 5000000 and i < 6000000:
        return 14
    elif i >= 6000000 and i < 7000000:
        return 16
    elif i >= 7000000 and i < 8000000:
        return 18
    elif i >= 8000000 and i < 9000000:
        return 20
    elif i >= 9000000 and i < 10000000:
        return 22
    else:
        return 24

# def score_change(i):
#     if i < 1000000:
#         return 1
#     elif i < 2000000 and i >= 1000000:
#         return 2
#     elif i < 3000000 and i >= 2000000:
#         return 3
#     elif i < 4000000 and i >= 3000000:
#         return 4
#     elif i < 5000000 and i >= 4000000:
#         return 5
#     elif i < 6000000 and i >= 5000000:
#         return 6
#     elif i < 7000000 and i >= 6000000:
#         return 7
#     elif i < 8000000 and i >= 7000000:
#         return 8
#     elif i < 9000000 and i >= 8000000:
#         return 9
#     else:
#         return 10
#
cate_set["final_audience"] = cate_set["final_audience"].apply(resizeValue)
#print(cate_set["final_audience"])

#관객수랑 각 애트리뷰트 묶음
d1 = cate_set[["director","final_audience"]]
o1 = cate_set[["openDt","final_audience"]]
p1 = cate_set[["prdtYear","final_audience"]]
n1 = cate_set[["repNationNm","final_audience"]]
g1 = cate_set[["repGenreNm","final_audience"]]
s1 = cate_set[["showTm","final_audience"]]
w1 = cate_set[["watchGradeNm","final_audience"]]
c1 = cate_set[["distributor","final_audience"]]

# 여기서 점수들을 매김. 최종관객수(1~10점)를 따져서 평균을 매김. ex) 홍지영 감독의 영화는 2점 2점 1점 총 3개 -> final_audience = 1.666667
d2 = d1.groupby("director").agg({"final_audience": np.sum})
o2 = o1.groupby("openDt").agg({"final_audience": np.mean})
p2 = p1.groupby("prdtYear").agg({"final_audience": np.mean})
n2 = n1.groupby("repNationNm").agg({"final_audience": np.mean})
g2 = g1.groupby("repGenreNm").agg({"final_audience": np.mean})
s2 = s1.groupby("showTm").agg({"final_audience": np.mean})
w2 = w1.groupby("watchGradeNm").agg({"final_audience": np.mean})
c2 = c1.groupby("distributor").agg({"final_audience": np.mean})


d2 = d2.reset_index()
o2 = o2.reset_index()
p2 = p2.reset_index()
n2 = n2.reset_index()
g2 = g2.reset_index()
s2 = s2.reset_index()
w2 = w2.reset_index()
c2 = c2.reset_index()

# d2 = d2.rename(columns={"final_audience":"director_score"})
# o2 = o2.rename(columns={"final_audience":"openDt_score"})
# p2 = p2.rename(columns={"final_audience":"prdtYear_score"})
# n2 = n2.rename(columns={"final_audience":"repNationNm_score"})
# g2 = g2.rename(columns={"final_audience":"repGenreNm_score"})
# s2 = s2.rename(columns={"final_audience":"showTm_score"})
# w2 = w2.rename(columns={"final_audience":"watchGradeNm_score"})
# c2 = c2.rename(columns={"final_audience":"distributor_score"})

# print(o2)
# openDt_score_set = o2[["openDt","openDt_score"]]
# prdtYear_score_set = p2[["prdtYear","prdtYear_score"]]
# nation_score_set = n2[["repNationNm","repNationNm_score"]]
# genre_score_set = g2[["repGenreNm","repGenreNm_score"]]
# showTm_score_set = s2[["showTm","showTm_score"]]
# watchGrade_score_set = w2[["watchGradeNm","watchGradeNm_score"]]
# director_score_set = d2[["director","director_score"]]
# distributor_score_set = c2[["distributor","distributor_score"]]
# # print(o2)
# #
# openDt_score_set.to_csv(os.getcwd() + '/../score/openDt_score.csv',encoding="utf-8")          # 개봉날짜
# prdtYear_score_set.to_csv(os.getcwd() + '/../score/prdtYear_score.csv',encoding="utf-8")      # 제작연도
# nation_score_set.to_csv(os.getcwd() + '/../score/nation_score.csv',encoding="utf-8")          # 국가
# genre_score_set.to_csv(os.getcwd() + '/../score/genre_score.csv',encoding="utf-8")            # 장르
# showTm_score_set.to_csv(os.getcwd() + '/../score/showTm_score.csv',encoding="utf-8")          # 쇼타임
# watchGrade_score_set.to_csv(os.getcwd() + '/../score/watchGrade_score.csv',encoding="utf-8")  # 등급
# director_score_set.to_csv(os.getcwd() + '/../score/director_score.csv',encoding="utf-8")      # 감독
# distributor_score_set.to_csv(os.getcwd() + '/../score/distributor_score.csv',encoding="utf-8")# 배급사

#합쳐버리기~ 기준있음 //각 애트리뷰트별로 정렬하면서 inner join 했으므로, 동일한 컬럼인 final_audience는 x, y로 나뉨
#그래서 final_audience_y를 score로 이름 변경
d3 = cate_set.merge(d2,how='inner',on='director')
o3 = cate_set.merge(o2,how='inner',on='openDt')
p3 = cate_set.merge(p2,how='inner',on='prdtYear')
n3 = cate_set.merge(n2,how='inner',on='repNationNm')
g3 = cate_set.merge(g2,how='inner',on='repGenreNm')
s3 = cate_set.merge(s2,how='inner',on='showTm')
w3 = cate_set.merge(w2,how='inner',on='watchGradeNm')
c3 = cate_set.merge(c2,how='inner',on='distributor')

d3 = d3.rename(columns={"final_audience_y":"director_score"})
o3 = o3.rename(columns={"final_audience_y":"openDt_score"})
p3 = p3.rename(columns={"final_audience_y":"prdtYear_score"})
n3 = n3.rename(columns={"final_audience_y":"repNationNm_score"})
g3 = g3.rename(columns={"final_audience_y":"repGenreNm_score"})
s3 = s3.rename(columns={"final_audience_y":"showTm_score"})
w3 = w3.rename(columns={"final_audience_y":"watchGradeNm_score"})
c3 = c3.rename(columns={"final_audience_y":"distributor_score"})
#

d3 = d3[["movieCd","director_score"]]
o3 = o3[["movieCd","openDt_score"]]
p3 = p3[["movieCd","prdtYear_score"]]
n3 = n3[["movieCd","repNationNm_score"]]
g3 = g3[["movieCd","repGenreNm_score"]]
s3 = s3[["movieCd","showTm_score"]]
w3 = w3[["movieCd","watchGradeNm_score"]]
c3 = c3[["movieCd","distributor_score"]]


#
a11 = cate_set[["movieCd","actor1","final_audience"]]
a12 = cate_set[["movieCd","actor2","final_audience"]]
a13 = cate_set[["movieCd","actor3","final_audience"]]
a14 = cate_set[["movieCd","actor4","final_audience"]]
a15 = cate_set[["movieCd","actor5","final_audience"]]
a16 = cate_set[["movieCd","actor6","final_audience"]]
a11.to_csv("a11.csv",encoding="utf-8")


# 여기서 배우별 점수들을 매김. 최종관객수(1~10점)를 따져서 합을 매김. ex) 홍지영 감독의 영화는 2점 2점 1점 총 3개 -> final_audience = 5
a21 = a11.groupby("actor1").agg({"final_audience": np.sum})
a22 = a12.groupby("actor2").agg({"final_audience": np.sum})
a23 = a13.groupby("actor3").agg({"final_audience": np.sum})
a24 = a14.groupby("actor4").agg({"final_audience": np.sum})
a25 = a15.groupby("actor5").agg({"final_audience": np.sum})
a26 = a16.groupby("actor6").agg({"final_audience": np.sum})
                        ...
actor = actor.merge(actor2,how="outer",on="movieCd")
actor["actor_score"] = actor["final_audience_y_x_x"]+ actor["final_audience_y_y_x"] + actor["final_audience_y_x"]+\
                        actor["final_audience_y_x_y"]+ actor["final_audience_y_y_y"] + actor["final_audience_y_y"]

actor = actor[["movieCd","actor_score"]]
actor = actor.fillna(0)
'''
# #
a21 = a21.reset_index()
a22 = a22.reset_index()
a23 = a23.reset_index()
a24 = a24.reset_index()
a25 = a25.reset_index()
a26 = a26.reset_index()
a21.to_csv("a21.csv",encoding="utf-8")
#actor를 기준으로 outer join함
a31 = cate_set.merge(a21,how='outer',on='actor1')
a32 = cate_set.merge(a22,how='outer',on='actor2')
a33 = cate_set.merge(a23,how='outer',on='actor3')
a34 = cate_set.merge(a24,how='outer',on='actor4')
a35 = cate_set.merge(a25,how='outer',on='actor5')
a36 = cate_set.merge(a26,how='outer',on='actor6')


#각각 actor/final_audience_y 값을 가짐. 첫번째배우 = actor, 첫번째 배우 점수 = final_audience_y
a31 = a31.rename(columns = {"actor1":"actor"})
a32 = a32.rename(columns = {"actor2":"actor"})
a33 = a33.rename(columns = {"actor3":"actor"})
a34 = a34.rename(columns = {"actor4":"actor"})
a35 = a35.rename(columns = {"actor5":"actor"})
a36 = a36.rename(columns = {"actor6":"actor"})

#각 actor 와 final_audience_y를 가져옴
a41 = a31[["actor","final_audience_y"]]
a42 = a32[["actor","final_audience_y"]]
a43 = a33[["actor","final_audience_y"]]
a44 = a34[["actor","final_audience_y"]]
a45 = a35[["actor","final_audience_y"]]
a46 = a36[["actor","final_audience_y"]]
'''

#합침
actor = pd.concat([a41,a42,a43,a44,a45,a46])
print(actor)
actor = actor.groupby("actor").agg({"final_audience_y":np.sum})

actor = actor.reset_index()

# actor.ix[0,'final_audience_y'] = 2

actor.to_csv(os.getcwd() + '/../score/actor_score.csv',encoding="utf-8")            # 배우          # 배우

# a31 = a31[["movieCd","final_audience_y"]]
# a32 = a32[["movieCd","final_audience_y"]]
# a33 = a33[["movieCd","final_audience_y"]]
# a34 = a34[["movieCd","final_audience_y"]]
# a35 = a35[["movieCd","final_audience_y"]]
# a36 = a36[["movieCd","final_audience_y"]]

# actor = a31.merge(a32,how="outer",on="movieCd")
# actor = actor.merge(a33,how="outer",on="movieCd")
# actor2 = a34.merge(a35,how="outer",on="movieCd")
# actor2 = actor2.merge(a36,how="outer",on="movieCd")

actor = actor.merge(actor2,how="outer",on="movieCd")
actor["actor_score"] = actor["final_audience_y_x_x"]+ actor["final_audience_y_y_x"] + actor["final_audience_y_x"]+\
                        actor["final_audience_y_x_y"]+ actor["final_audience_y_y_y"] + actor["final_audience_y_y"]

actor = actor[["movieCd","actor_score"]]
actor = actor.fillna(0)
#actor.to_csv("actortest.csv",encoding="utf-8")
#
value_data_set = data_set.drop(['director', 'openDt', 'prdtYear', 'repNationNm', 'repGenreNm', 'showTm', 'watchGradeNm',
                                'typeNm', 'actor1', 'actor2', 'actor3','actor4','actor5','actor6', 'distributor',
                                'companyCd','producedCompany'],axis=1)
#
dd_set = value_data_set.merge(d3,how="outer",on="movieCd")
dd_set = dd_set.merge(o3,how="outer",on="movieCd")
dd_set = dd_set.merge(p3,how="outer",on="movieCd")
dd_set = dd_set.merge(n3,how="outer",on="movieCd")
dd_set = dd_set.merge(g3,how="outer",on="movieCd")
dd_set = dd_set.merge(s3,how="outer",on="movieCd")
dd_set = dd_set.merge(w3,how="outer",on="movieCd")
dd_set = dd_set.merge(c3,how="outer",on="movieCd")
dd_set = dd_set.merge(actor,how="outer",on="movieCd")

#dd_set.to_csv("dd_setTest.csv",encoding="utf-8")
dd_set["director_score"] = round(dd_set["director_score"],1)
dd_set["openDt_score"] = round(dd_set["openDt_score"],1)
dd_set["prdtYear_score"] = round(dd_set["prdtYear_score"],1)
dd_set["repNationNm_score"] = round(dd_set["repNationNm_score"],1)
dd_set["repGenreNm_score"] = round(dd_set["repGenreNm_score"],1)
dd_set["showTm_score"] = round(dd_set["showTm_score"],1)
dd_set["watchGradeNm_score"] = round(dd_set["watchGradeNm_score"],1)
dd_set["distributor_score"] = round(dd_set["distributor_score"],1)
dd_set["actor_score"] = round(dd_set["actor_score"],1)
# #
# dd_set = dd_set.drop(["movieCd","movieNm"],axis=1)
#
dd_set = dd_set.drop(["movieNm"], axis=1)
dd_set = dd_set.fillna(0)
#
print(dd_set.columns)
dd_set = dd_set[['movieCd','director_score','openDt_score', 'prdtYear_score', 'repNationNm_score', 'repGenreNm_score',
                 'showTm_score', 'watchGradeNm_score', 'distributor_score', 'actor_score','starScore', 'userCount',
                 'previous_screen','screen_D1', 'screen_D2', 'screen_D3','screen_D4', 'screen_D5', 'screen_D6',
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
                 'audience_D25', 'audience_D26', 'audience_D27', 'audience_D28', 'final_audience' ]]
dd_set["starScore"] = dd_set["starScore"].replace(" ",0)
dd_set["userCount"] = dd_set["userCount"].replace(" ",0)
# dd_set = dd_set[dd_set["final_audience"]>50000].reset_index(drop=True)
dd_set.to_csv(os.getcwd() + "/../checkcheck/value_data.csv", encoding="utf-8")
# # # '''
# # # 스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에 선형 변환을 적용하여
# # # 전체 자료의 분포를 평균 0, 분산 1이 되도록 만드는 과정이다.
# # # 스케일링은 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지하고
# # # 독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜
# # # 최적화 과정에서의 안정성 및 수렴 속도를 향상시킨다.
# # #  scale(X): 기본 스케일. 평균과 표준편차 사용
# # #  robust_scale(X): 중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화
# # # '''
# # # # dd_set.to_csv(os.getcwd()+"/../data/21_non_scaled_value_data.csv",encoding="utf-8")
# # # from sklearn.preprocessing import StandardScaler
# # # data_set = pd.read_csv(os.getcwd() + '/../data/26_non_scaled_value_with_test_value_data.csv',index_col =0)
# # # movieCd = data_set["movieCd"]
# # # dd_set = data_set.drop(["movieCd"],axis=1)
# # # dataset = dd_set.ix[:,:-1]
# # # target = dd_set.ix[:,-1]
# # #
# # # scaler = StandardScaler(with_mean=False)
# # # dataset_scaled = scaler.fit_transform(dataset) # = fit / transform
# # #
# # # dataset_forDF= pd.DataFrame(dataset_scaled,columns=dataset.columns) # 스케일링된 데이터셋
# # # target_forDF= pd.DataFrame(dd_set.ix[:,-1],columns=["final_audience"]) # 스케일링 안된 그냥 타겟
# # #
# # # scaled_DF = pd.concat([movieCd, dataset_forDF, target_forDF], axis=1)
# # # scaled_DF.to_csv(os.getcwd()+"/../data/27_scaled_value_with_test_value_data.csv",encoding="utf-8")
# # #
# # #
# # # # 애트리뷰트 순서 바꿔서 저장함 openDt 값도 정상적으로 나옴
# # # # string으로 되어있던 star Score, user Count는 empty(" ")값 때문이었음. (null, nan , None 아님) 비어있던 공간 없애니
# # # # 자동으로 타입 변경됨
# # # # 5만명 이상으로 추림 // 1~5만 사이는 관객수가 너무 적어서 예측 변동폭이 너무 큼.
# # # # -> 얼마 차이 안나지만 예측률을 크게 떨어뜨리는 요소