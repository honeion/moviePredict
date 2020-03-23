# -*- coding:utf-8 -*-
import pandas as pd
import os
import requests
from properties import KEY
movieInfoDF = pd.read_csv(os.getcwd() + '/../data/02_movie_data_info.csv', engine='python')

key = KEY
keyCount = 0
movieCd = movieInfoDF["movieCd"]

# 영화상세정보 가져오기
# 파라미터로 key, 아이템당 조회 페이지, 영화이름, 조회시작 연도, 조회종료 연도를 사용
def getMovieDetail(movieCd):
    try:
        global keyCount
        url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json"
        params = {"key": key[keyCount], "movieCd": movieCd}
        r = requests.get(url, params=params)

        if (r.json()['movieInfoResult']['movieInfo']['movieCd'] == None):
            print("*****데이터 없음. code :", movieCd)  # 데이터가 없는 경우
            return None

    except Exception as ex:
        print("에러종류: ", ex, "영화코드: ", movieCd)


    return r.json()


def makeMovieDetailDF(movieCd):
    '''
    showTm 문자열 상영시간을 출력합니다.              V
    actors 문자열 배우를 나타냅니다.                V
    peopleNm 문자열 배우명을 출력합니다.            V
    watchGradeNm 문자열 관람등급 명칭을 출력합니다.    V
    '''
    # 배우(6명), 감독(2명), 영화사(1번째 제작사, 1번째 배급사)가 같은 방식으로 매겨져야하는데.

    global keyCount
    movieDetailDF = pd.DataFrame(columns=["movieCd", "showTm", "actor1", "actor2", "actor3",
                                          "actor4", "actor5", "actor6", "watchGradeNm", "producedCompany",
                                          "distributor"])
    for code in movieCd:
        try:
            dataList = getMovieDetail(code)['movieInfoResult']['movieInfo']
            actors = []
            for i in range(len(dataList['actors'])):
                if i > 5:
                    break
                actors.append(dataList["actors"][i]["peopleNm"])

            for i in range(len(actors), 6):
                actors.append(" ")

            count = 0
            pflg = False
            dflg = False
            proCompany = " "
            disCompany = " "

            watchGrade = dataList["audits"][0]["watchGradeNm"]
            for count in range(0, len(dataList['companys'])):
                if (not pflg and dataList['companys'][count]['companyPartNm'] == '제작사'):
                    proCompany = dataList['companys'][count]['companyNm']
                    pflg = True

                if (not dflg and dataList['companys'][count]['companyPartNm'] == '배급사'):
                    disCompany = dataList['companys'][count]['companyNm']
                    dflg = True

            movieDetailDF.loc[len(movieDetailDF)] = [
                dataList["movieCd"],
                dataList["showTm"],
                actors[0],
                actors[1],
                actors[2],
                actors[3],
                actors[4],
                actors[5],
                watchGrade,
                proCompany,
                disCompany
            ]

        except KeyError:
            print("KeyError 영화코드: ", code)
        except Exception as ex:
            print("에러종류: ", ex, "코드", code)

    return movieDetailDF

movieDetailDF = makeMovieDetailDF(movieCd)

# print(movieDetailDF)
# # 새로운 csv 파일로 저장
movieDetailDF.to_csv(os.getcwd() + '/../data/03_movie_data_etc.csv', encoding='utf-8')

movieInfoDF = pd.read_csv(os.getcwd() + '/../data/02_movie_data_info.csv', index_col=0, engine='python', encoding='utf-8')

movieEctDF = pd.read_csv(os.getcwd() + '/../data/03_movie_data_etc.csv', index_col=0, engine='python', encoding='utf-8')

mergedMovieDf = pd.merge(movieInfoDF, movieEctDF)
mergedMovieDf = mergedMovieDf.reset_index(drop=True)
mergedMovieDf.to_csv(os.getcwd() + '/../data/04_movie_data_merged.csv', encoding='utf-8')