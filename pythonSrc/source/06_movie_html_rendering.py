# html 내용 합침
# -*- coding:utf-8 -*-
import pandas as pd
import os
import re

def makeList(statistical_Info):
    for __ in range(29 - len(statistical_Info["screen"])):
        statistical_Info = statistical_Info.append([{"screen": 0, "show": 0, "audience": 0}], ignore_index=True)

    return statistical_Info

# statisticalHtmlMovieDataCrawling.py 모듈에서 가져온 html파일을 데이터프레임으로
# 변형해서 csv파일에 저장한뒤에  전체 데이터 통합
# 영화진흥위원회 홈페이지에서 영화코드로 검색한 뒤 통계정보 추출

# 영화 이름 및 개봉날짜 저장
movieList = pd.read_csv(os.getcwd() + '/../data/06_movie_data_star_merged.csv', index_col=0)
movieNm = movieList['movieNm']
movieCd = movieList['movieCd']
movieNm2 = []  # html 파일 이름이 .:부분이 제거되었으므로 똑같이 만들어 이용
for i in range(0, len(movieNm)):
    movieNm2.append(re.sub('[=#/?:$\\n{}()]', '', movieNm[i]))

openDt = movieList['openDt']

try:
    # 새롭게 변경된 데이터를 속성별로 나누어 저장할 데이터프레임 생성
    statistical_Info_Df = pd.DataFrame(columns=["movieCd",  # 영화코드 이름
                                                "previous_screen",  # 개봉전 스크린수
                                                "screen_D1", "screen_D2", "screen_D3", "screen_D4",  # 개봉1~14일간 스크린수
                                                "screen_D5", "screen_D6", "screen_D7", "screen_D8",
                                                "screen_D9", "screen_D10", "screen_D11", "screen_D12",
                                                "screen_D13", "screen_D14",
                                                "screen_D15", "screen_D16", "screen_D17", "screen_D18",  # 개봉14~28일간 스크린수
                                                "screen_D19", "screen_D20", "screen_D21", "screen_D22",
                                                "screen_D23", "screen_D24", "screen_D25", "screen_D26",
                                                "screen_D27", "screen_D28",
                                                "previous_show",  # 개봉전 상영횟수
                                                "show_D1", "show_D2", "show_D3", "show_D4",  # 개봉1~14일간 상영횟수
                                                "show_D5", "show_D6", "show_D7", "show_D8",
                                                "show_D9", "show_D10", "show_D11", "show_D12",
                                                "show_D13", "show_D14",
                                                "show_D15", "show_D16", "show_D17", "show_D18",  # 개봉14~28일간 상영횟수
                                                "show_D19", "show_D20", "show_D21", "show_D22",
                                                "show_D23", "show_D24", "show_D25", "show_D26",
                                                "show_D27", "show_D28",
                                                "previous_audience",  # 개봉전 관객수
                                                "audience_D1", "audience_D2", "audience_D3", "audience_D4",# 개봉1~14일간 관객수
                                                "audience_D5", "audience_D6", "audience_D7", "audience_D8",
                                                "audience_D9", "audience_D10", "audience_D11", "audience_D12",
                                                "audience_D13", "audience_D14",
                                                "audience_D15", "audience_D16", "audience_D17", "audience_D18",# 개봉14~28일간 관객수
                                                "audience_D19", "audience_D20", "audience_D21", "audience_D22",
                                                "audience_D23", "audience_D24", "audience_D25", "audience_D26",
                                                "audience_D27", "audience_D28",
                                                "final_audience"                                               #최종 관객수

                                                ])
    # html 파일 읽어드림
    for index in range(0,len(movieCd)):
        try:
            path = "D:/workspace_python/modifiedMovieProject/collected_html/" + movieNm2[index] + "_"+movieCd[index]+".html"  # 경로는 데이터폴더로 하자고 모아서
            print(movieNm2[index], movieCd[index])
            movie_Html_Info = pd.read_html(path)
            # html파일을 데이터프레임으로 변환 //누락값을 제거하는 dropna axis = 0이면 누락값 행을 제거, 1이면 포함된 열 제거
            movie_Html_Info = movie_Html_Info[0].dropna(axis=0)  # 이것만으로 데이터프레임됨 [0]만 되네
            #list[:]는 list의 얕은 복사로, 둘이 동일한 값. 통채 복사는 안먹힘. 강제로 형변환 된듯

            # 변환시킨 데이터프레임에서 개봉이후 14일까지 데이터부분 추출
            # 처음 날짜데이터는 yy-mm-dd로 표기되어있어서 yymmdd로 변환
            temp_Date_List = []  # 빈 리스트 생성
            for x in movie_Html_Info["날짜"]:
                # 빈 리스트에 날짜값 변형해서 삽입
                temp_Date_List.append(x.replace('-', ''))
            # 데이터프레임 날짜데이터 수정
            # 개봉날짜랑 비교해서 개봉이전+14일간의 데이터
            # 임시 데이터프레임에 컬럼명 바꿔서 저장

            tempoDf = pd.DataFrame({"date": temp_Date_List,
                                    "screen": movie_Html_Info["스크린수"],
                                    "show": movie_Html_Info["상영횟수"],
                                    "audience": movie_Html_Info["관객수"]})
            # 개봉전까지가 어디인지 확인해줄 변수
            # 개봉전에 상영한 날짜가 여러개일수도 없을 수도 있음
            count = 0
            # 받아온 개봉날짜보다 이전에 상영했던 날이 있을때마다 count증가
            for i in range(0, len(tempoDf['date'])):
                if (int(tempoDf['date'][i]) < int(openDt[index])):
                    count += 1
            print("count:", count)
            # 통계정보는 개봉후 28일까지 정보만 필요하므로, 개봉전+28일데이터 수집 및 컬럼 순서 변경
            statistical_Info = tempoDf.head(count+28)
            # 개봉전 데이터들 한개의 데이터로 결합
            previous_List = []
            screen = 0
            show = 0
            audience = 0

            for j in range(0, count):
                screen += statistical_Info.ix[j]['screen']
                show += statistical_Info.ix[j]['show']
                audience += statistical_Info.ix[j]['audience']

            previous_List = ["previous", screen, show, audience]
            # 결합된 데이터를 기존의 데이터 삭제후 첫행에 추가
            # count는 최소 1개이므로 개봉전 데이터가 1개면 바로 변경
            if (count == 0):
               # print("야")
                statistical_Info.loc[-1] = previous_List
                statistical_Info.index = statistical_Info.index + 1
                statistical_Info = statistical_Info.sort_index()
            else:
                for i in range(0, count-1):
                    # ex) count = 3이면 2개까지 0,1을 삭제후 2번에 합쳐진 데이터를 새로 넣고 인덱스를 리셋
                    statistical_Info = statistical_Info.drop([i])
                tempDf = statistical_Info
                tempDf = tempDf.reset_index()
                statistical_Info.loc[count-1] = previous_List
                statistical_Info = statistical_Info.reset_index(drop=True)

            statistical_Info = makeList(statistical_Info)
            temp_for_final_audience = pd.DataFrame({"final_audience": movie_Html_Info["누적관객수"].astype(int)})
            statistical_Info_final = temp_for_final_audience.tail(1)

            # 속성에 따라 데이터 삽입 = 영화하나에 대한 데이터 한 줄
            statistical_Info_Df.loc[len(statistical_Info_Df)] = [
                movieCd[index],
                *statistical_Info["screen"],
                *statistical_Info["show"],
                *statistical_Info["audience"],
                statistical_Info_final.get_values()[0][0] # np.ndarray라서 이렇게 꺼냄
            ]

        except :
            index += 1
            print("에바임")

except:
    print("영화통계정보 오류")


# 데이터프레임을 csv파일로 저장
statistical_Info_Df.to_csv(os.getcwd() + '/../data/07_movie_Statistical_Info.csv', encoding='utf-8')

# 수집된 통계정보를 영화정보와 평점데이터가 결합된 데이터에 추가
merged_MovieInfo_Star_Df = pd.read_csv(os.getcwd() + '/../data/06_movie_data_star_merged.csv', index_col=0)
statistical_Info_Df = pd.read_csv(os.getcwd() + '/../data/07_movie_Statistical_Info.csv', index_col=0)
# 최종데이터로 결합
final_Movie_Data_Df = pd.merge(merged_MovieInfo_Star_Df, statistical_Info_Df)
final_Movie_Data_Df = final_Movie_Data_Df.drop_duplicates("movieCd")  # 중복값 제거
final_Movie_Data_Df = final_Movie_Data_Df.reset_index(drop=True)  # 인덱스 갱신
final_Movie_Data_Df.to_csv(os.getcwd() + '/../data/08_completed_MovieData.csv', encoding='utf-8')
# 영화기본+상세정보+별점데이터+통계정보 병합