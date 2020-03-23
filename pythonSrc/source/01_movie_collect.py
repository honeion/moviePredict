# -*- coding:utf-8 -*-
import pandas as pd
import os
import requests
from properties import KEY

# 영화진흥위원회에서 2008년 1월 ~ 2018년 4월까지의  관객 1만명 이상의 데이터 수집
# 서울매출, 서울관객 속성을 제외하고 csv 파일 형태로 저장
# 영화 데이터 프레임으로 저장된 csv 파일을 호출

movieDF = pd.read_csv(os.getcwd()+'/../data/01_movie_data.csv',engine='python')

# 영화데이터 프레임에서 순번을 삭제하고 인덱스를 다시 매긴뒤, 개봉일 데이터 '-'삭제
del movieDF["순번"]
movieDF = movieDF.reset_index(drop=True)
movieDF["개봉일"] = movieDF["개봉일"].str.replace("-","")
# movieDF["전국관객수"] = movieDF["전국관객수"].replace(",","")
# movieDF["전국관객수"] = movieDF["전국관객수"].astype(int)
# movieDF1800 = movieDF[movieDF["전국관객수"]>=50000].reset_index(drop=True)

# 가져온 영화 데이터 프레임으로 부터 영화명, 개봉일 변수에 각각 저장
movieNm = movieDF["영화명"]
openDt = movieDF["개봉일"]

# 영화 개봉일로 부터 개봉연도만 추출
openStartDt=[]
for index in range(0,len(openDt)):
    openStartDt.append(openDt[index][0:4])
openStartDt = pd.Series(openStartDt)
openEndDt = openStartDt

# 영화진흥위원회 오픈API 사용
# 키를 받아 영화코드,영화제목,감독,제작연도,개봉연도,영화유형,대표국가,대표장르,영화사코드 수집
# 영화 이름을 통해서 조회, 영화에 대한 정보 조회중 1개의 데이터가 조회를 5번 이상할 시에 넘김
# 영화 이름을 통해서 조회하므로 영화에 대한 정보가 여러개 출력됨

key = KEY

# 영화리스트 가져오기
# 파라미터로 key, 아이템당 조회 페이지, 영화이름, 조회시작 연도, 조회종료 연도를 사용
def getMovieList(movieNm,openStartDt,openEndDt):
    for index in range(0,len(key)):
        try:
            url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json"
            params = {"key": key[index], "itemPerPage": 5, "movieNm": movieNm,
                      "openStartDt":openStartDt,"openEndDt":openEndDt}
            r = requests.get(url, params=params)
            return r.json()
        except:
            continue
        
#영화 목록 및 기본정보를 갖도록 영화진흥위원회 open API로 수집 후 데이터 프레임 구성
def makeMovieInfo(movieNm):
    
    # 영화진흥위원회 오픈API 영화목록 실제 요청구조
    '''
    # key(필수)	    발급받은키 값
    # curPage       현재 페이지(default : “1”)
    # itemPerPage   결과 ROW 의 개수(default :"10”)
    # movieNm       영화명으로 조회(UTF-8 인코딩)
    # directorNm    감독명으로 조회
    # openStartDt   YYYY형식의 조회시작 개봉연도
    # openEndDt	    YYYY형식의 조회종료 개봉연도
    # prdtStartYear YYYY형식의 조회시작 제작연도를 입력
    # prdtEndYear   YYYY형식의 조회종료 제작연도를 입력
    # repNationCd   N개의 국적으로 조회
    # movieTypeCd   N개의 영화유형코드

    '''
    # 영화진흥위원회 오픈API 영화목록 실제 응답구조
    '''
    # movieCd     영화코드를 출력   # movieNm     영화명(국문)     # movieNmEn   영화명(영문)  # prdtYear    제작연도
    # openDt      개봉날짜          # typeNm      영화유형         # prdtStatNm  제작상태      # nationAlt   제작국가(전체)
    # genreAlt    영화장르(전체)    # repNationNm 대표 제작국가명  # repGenreNm  대표 장르명   # directors   영화감독
    # peopleNm    영화감독명        # companys    제작사           # companyCd   제작사 코드   # companyNm   제작사명
    #
    # 조회된 영화목록은 {'movieListResult':
    #                             {'totCnt':, 'source':,
    #                              'movieList':
    #                                         [{'movieCd':, 'movieNm':, 'movieNmEn':,
    #                                          'prdtYear':, 'openDt':, 'typeNm':, 'prdtStatNm':, 'nationAlt':,
    #                                          'genreAlt':, 'repNationNm':, 'repGenreNm':,
    #                                          'directors': [{'peopleNm':}], 'companys': [{'companyCd':, 'companyNm':}]
    #                                                                                     과 같은 형태의 정보로 사전구조의 json 형식
    '''
    # 필요정보만 추출하기 위해 데이터프레임 작성
    # 영화코드, 영화명, 감독1, 감독2, 제작연도, 개봉날짜, 영화유형, 대표국가, 대표장르, 영화사코드
    movieInfoDF = pd.DataFrame(columns=["movieCd", "movieNm", "director1","director2", "prdtYear", "openDt", "typeNm","repNationNm", "repGenreNm", "companyCd"])
    
    # 개봉연도, 개봉날짜를 위한 카운트 변수
    count = -1
    for names in movieNm:  # 영화이름 목록 이용
        count += 1
        
        # dataList = 위 사전구조에서 movieList내의 목록들을 가짐
        for dataList in getMovieList(names,openStartDt[count],openEndDt[count])['movieListResult']['movieList']:  
 
            try:
                # 개봉연도는 동일하나 이름이 비슷한 경우가 존재하므로, 개봉날짜의 동일함을 비교
                if (str(openDt[count]) == dataList['openDt'] and str(movieNm[count]) == dataList['movieNm']):  

                    # 영화감독이  2이상이면 첫번째, 두번째 감독으로 선택
                    if len(dataList['directors']) >= 2:  
                        director1 = dataList["directors"][0]["peopleNm"]
                        director2 = dataList["directors"][1]["peopleNm"]
                    # 영화감독이 1명이면 첫번째 감독
                    elif len(dataList['directors']) == 1:
                        director1 = dataList["directors"][0]["peopleNm"]
                        director2 = " "
                    # 감독이 없는경우
                    else:  
                        director1 = " "
                        director2 = " "
                        
                    # companyCd  영화사 코드로 영화사명(companyNm, companyPartNm)을 추출할 수 있으므로 코드만 수집
                    # 제작사가 1이상이면 첫번째회사로 선택
                    if len(dataList['companys']) >= 1:  
                        companyCd = dataList["companys"][0]["companyCd"]
                    else:
                        companyCd = " "

                # 개봉연도는 동일, 개봉날짜는 다른 경우 다음 names를 이용하는 상위 for문으로 넘어감
                else:
                    continue
                    
                #print("삽입정보-영화코드:", dataList["movieCd"], " 영화명:", dataList["movieNm"], " 개봉날짜:", dataList["openDt"],"제작사코드:", companyCd, "영화감독1:", director1, "영화감독2:", director2)

                # 순서에 맞게 수집한 데이터 데이터프레임에 삽입
                movieInfoDF.loc[len(movieInfoDF)] = [  
                    dataList["movieCd"],    #영화코드
                    dataList["movieNm"],    #영화명
                    director1,              #감독1
                    director2,              #감독2
                    dataList["prdtYear"],   #제작연도
                    dataList["openDt"],     #개봉일
                    dataList["typeNm"],     #영화유형
                    dataList["repNationNm"],#대표국가
                    dataList["repGenreNm"], #대표장르
                    companyCd               #영화사코드
                ]
            except:
                print("영화기본정보수집 오류")
                return movieInfoDF

    return movieInfoDF

# # 영화 목록, 기본 정보들을 갖는 데이터 프레임 생성
movieInfoDF = makeMovieInfo(movieNm)
print(movieInfoDF)
# # 새로운 csv 파일로 저장
# movieInfoDF.to_csv(os.getcwd()+'/../data/02_movie_data_info.csv',encoding='utf-8')

