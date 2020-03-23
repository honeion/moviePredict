# -*-coding:utf-8 -*-
import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.alert import Alert
import os
import re
from properties import HTMLDownloadPATH, ChromeDriverPATH

# 영화진흥위원회 홈페이지에서 데이터 추출
# 가져온 영화정보뿐만아니라 영화코드를 이용해 DB검색을 통해 통계정보에 접근
# 웹페이지상에는 개봉후 10일까지의 데이터뿐이므로, 엑셀파일을 다운로드 받음
# 엑셀파일은 html로 작성되었고, xls파일이라 전부 영화명에따라 html로 변환하는 모듈로
# 다른 모듈을 통해 html을 데이터프레임형태로 바꾸기 위한 디딤 모듈임

# 다운로드는 오늘날짜에 따라 지어지는 일자별 데이터-실제날짜 이런식으므로 이름을 바꾸어주는 함수이용
def nameChange(index):
    os.chdir(HTMLDownloadPATH)  # 현재 작업디렉토리를 변환해서
    for i in range(0, len(os.listdir())):
        if "일자별" in os.listdir()[i]:
            fileNm = os.listdir()[i]
            os.rename(fileNm, movieNm2[index-2001]+'_'+movieCd[index]+ ".html")  # 이름과 확장명을 바꿔줌


# 영화목록 불러오기
movieList = pd.read_csv(os.getcwd() + '/../data/06_movie_data_star_merged.csv', index_col=0)
# 영화코드,이름 저장
movieCd = movieList['movieCd']
movieNm = movieList['movieNm']

movieNm2 = []  # html 파일로 저장하기위해서는 :.등의 문자가 이름에 있으면 안됨

for i in range(0, len(movieCd)):
    movieNm2.append(re.sub('[=#/?:$\\n{}()]', '', movieNm[i]))

#영화진흥위원회사이트를 크롤링을 하여 현재목록에 있는 모든 영화에 대한 파일을 다운로드 받음
#for index in range(0, len(movieList)):
for index in range(2000, len(movieCd)):
    try:
        # 다운로드 기본위치를 현재디렉토리 안에 html디렉토리로 설정
        chromeOptions = webdriver.ChromeOptions()
        prefs = {"download.default_directory":HTMLDownloadPATH}  # 다운로드 폴더 설정
        chromeOptions.add_experimental_option("prefs", prefs)
        # 드라이버 실행
        driver = webdriver.Chrome(ChromeDriverPATH, chrome_options=chromeOptions)

        # 진흥위원회 코드검색 화면 접속
        driver.get('http://www.kobis.or.kr/kobis/business/mast/mvie/searchUserMovCdList.do')

        # 영화 코드창에 입력하여 조회
        elem = driver.find_element_by_xpath(
            '//*[@id="content"]/form/fieldset/section/div[1]/table/tbody/tr[1]/td[2]/input')
        elem.send_keys(str(movieCd[index]))
        elem.submit()

        # 조회된 영화 클릭
        elem = driver.find_element_by_xpath('//*[@id="content"]/table/tbody/tr[1]/td[1]/a')
        elem.click()
        time.sleep(5)

        # 조회된 영화 정보에서 통계정보 클릭
        elem = driver.find_element_by_class_name('tab_layer')
        staticElem = elem.find_elements_by_tag_name('li')[1]
        staticElem.click()
        time.sleep(25)

        #잘못된 영화인지 구분
        Nm = driver.find_element_by_class_name('mtitle').text
        if Nm[0:len(movieNm[index])] != movieNm[index]:
            print("잘못된 영화를 가져옴!!!!!!!!"+"웹 이름 : " + Nm[0:len(movieNm[index])] + "  " + "실제 영화이름 : " + movieNm[index])
            # f = open("오류나나버려라")

        # 통계정보에서 엑셀 버튼을 클릭
        staticElem = driver.find_element_by_class_name("subt02")
        elem = staticElem.find_element_by_link_text("엑셀")
        elem.click()
        print("엑셀클릭")
        time.sleep(2)

        # 중간에 alert이 하나 뜨는데 다운로드 하겠나는 의미로 accept해줌
        Alert(driver).accept()
        time.sleep(5)
        driver.close()
        nameChange(index)

    except:
        print(movieNm[index])
        print("영화진흥위원회 파일 다운로드 오류")
    # 하나 다운로드받고 바로 이름과 확장명을 바꿔주는 식
