# 별점수집
# -*- coding: utf-8 -*-

import urllib.request
import time
import re
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from properties import ChromeDriverPATH

# 영화 정보가 삽입 되어있는 데이터프레임 읽어드림
movieInfoDF = pd.read_csv(os.getcwd()+'/../data/04_movie_data_merged.csv',encoding ='utf-8')
# 가져온 영화 데이터 프레임으로 부터 영화코드, 영화명, 개봉년도, 개봉일, 상영시간 변수에 각각 저장
movieCd = movieInfoDF["movieCd"]
movieNm = movieInfoDF["movieNm"]
openDt = movieInfoDF["openDt"]
showTm = movieInfoDF["showTm"]

naverDriver = webdriver.Chrome(ChromeDriverPATH) 
naverDriver.get('http://movie.naver.com/')
time.sleep(1)

def _checkCorrectMovie(check, comparisonData, j):
    Elem = naverDriver.find_element_by_xpath('//*[@id="old_content"]/ul[@class="search_list_1"]')
    # try except로 없는애들은 null값 주자.
    liList = Elem.find_elements_by_tag_name('li')

    for i in range(len(liList)):
        Elem = naverDriver.find_element_by_xpath('//*[@id="old_content"]/ul[@class="search_list_1"]')
        # try except로 없는애들은 null값 주자.
        liList = Elem.find_elements_by_tag_name('li')

        # print(type(liList[i].text), type(comparisonData))
        # 찾고자 하는 이름이 포함되어있으면
        if str(comparisonData) in liList[i].text:
            elem = liList[i].find_element_by_tag_name('a')
            elem.click()

            html = naverDriver.page_source
            soup = BeautifulSoup(html, 'lxml')

            try:
                opendts = soup.find('dl', class_="info_spec")
                opendt = opendts.find_all('span')[3]
            except:
                naverDriver.execute_script("window.history.go(-1)")
                continue
            str2 = opendt.get_text()
            str2 = str2.replace(".", "").replace("\n", "").replace("개봉", "")
            str2 = str2.strip()
            # 개봉일 다르면
            if str(openDt[j]) != str2:
                naverDriver.execute_script("window.history.go(-1)")
                continue

            else:
                # 리뷰에서 평점과 스코어 항목 가져오기
                netizen = soup.find('div', class_="score score_left").find('div', class_="star_score")
                ems = netizen.find_all('em')
                k = ""
                for em in ems:
                    k += em.get_text()
                # print("평점", k)
                user_count = soup.find('span', class_="user_count")
                # print(user_count)
                user_em = user_count.find('em').get_text()
                # print("참여인원", user_em)

                check = True
                return check, k, user_em

    return check, " ", " "


movieNaverDF = pd.DataFrame(columns=["movieCd", "starScore", "userCount"])

for i in range(len(movieCd)):
    # 웹에서 영화이름으로 검색
    naverElem = naverDriver.find_element_by_class_name('ipt_tx_srch')
    naverElem.send_keys(movieNm[i])
    time.sleep(1)
    naverElem = naverDriver.find_element_by_xpath('//*[@id="jSearchArea"]/div/button')
    naverElem.click()
    time.sleep(1)
    # 더 많은 영화 보기
    ElemMore = naverDriver.find_element_by_xpath('//*[@id="old_content"]/a[1]')
    emtext = ElemMore.text
    check = False
    if ("더 많은 영화 보기" == emtext.strip()):
        # 더 많은 영화 보기 클릭
        ElemMore.click()

        check, starScore, userCount = _checkCorrectMovie(check, showTm[i], i)
        # 상영시간으로 찾지 못했을 경우
        if not check:
            check, starScore, userCount = _checkCorrectMovie(check, movieNm[i], i)
            # 이름으로도 없으면 2페이지로 넘어가기
            if not check:
                try:
                    ElemMoreList = naverDriver.find_element_by_xpath('//*[@id="old_content"]/div[2]/table/tbody/tr')
                    elemmorelist = ElemMoreList.find_element_by_xpath('//td[2]')
                    #print(elemmorelist.text)
                    elemmorelist.click()
                    check, starScore, userCount = _checkCorrectMovie(check, showTm[i], i)
                    # 상영시간으로 찾지 못했을 경우
                    if not check:
                        check, starScore, userCount = _checkCorrectMovie(check, movieNm[i], i)
                except:
                    print("2페이지 없음")

    # 더 많은 영화 보기 없으면
    else:
        check, starScore, userCount = _checkCorrectMovie(check, showTm[i], i)
        # 상영시간으로 찾지 못했을 경우
        if not check:
            check, starScore, userCount = _checkCorrectMovie(check, movieNm[i], i)

    movieNaverDF.loc[len(movieNaverDF)] = [
        movieCd[i],
        starScore,
        userCount
    ]

# 새로운 csv 파일로 저장
movieNaverDF.to_csv(os.getcwd()+'/../data/05_movie_data_naverStarPoint.csv',encoding='utf-8')
movieInfoDF = pd.read_csv(os.getcwd()+'/../data/04movie_data_merged.csv',index_col=0,engine='python',encoding='utf-8')
movieNaverDF = pd.read_csv(os.getcwd()+'/../data/05_movie_data_naverStarPoint.csv',index_col=0,engine='python',encoding='utf-8')

# 새로 읽어 들인뒤 영화 코드를 기준으로 병합, 인덱스 리셋, 새로운 csv 파일로 저장
mergedMovieDf = pd.merge(movieInfoDF, movieNaverDF)
mergedMovieDf = mergedMovieDf.reset_index(drop=True)
mergedMovieDf.to_csv(os.getcwd()+'/../data/06_movie_data_star_merged.csv',encoding='utf-8')