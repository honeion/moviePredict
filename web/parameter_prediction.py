import sys
from urllib.parse import unquote
import pandas as pd
import os
import pickle
# pkl_filename = "pickle_randomForest.pkl"
# pkl_filename = "0_1.pkl"
# pkl_filename = "0_2.pkl"
pkl_filename = "clf.pkl"

kk = pd.DataFrame(columns=["openDt_score","prdtYear_score","repNationNm_score","repGenreNm_score",
                           "showTm_score","watchGradeNm_score","distributor_score","actor_score",
                           "starScore","userCount","previous_screen","previous_show","previous_audience"])

openDt_score_set = pd.read_csv(os.getcwd() + '/../data/score/openDt_score.csv',index_col =0)          # 개봉날짜
prdtYear_score_set = pd.read_csv(os.getcwd() + '/../data/score/prdtYear_score.csv',index_col =0)      # 제작연도
nation_score_set = pd.read_csv(os.getcwd() + '/../data/score/nation_score.csv',index_col =0)          # 국가
genre_score_set = pd.read_csv(os.getcwd() + '/../data/score/genre_score.csv',index_col =0)            # 장르
showTm_score_set = pd.read_csv(os.getcwd() + '/../data/score/showTm_score.csv',index_col =0)          # 쇼타임
watchGrade_score_set = pd.read_csv(os.getcwd() + '/../data/score/watchGrade_score.csv',index_col =0)  # 등급
director_score_set = pd.read_csv(os.getcwd() + '/../data/score/director_score.csv',index_col =0)      # 감독 --
actor_score_set = pd.read_csv(os.getcwd() + '/../data/score/actor_score.csv',index_col =0)            # 배우
distributor_score_set = pd.read_csv(os.getcwd() + '/../data/score/distributor_score.csv',index_col =0)# 배급사
actor_score_set = actor_score_set.rename(columns={"final_audience_y":"actor_score"})
# data = [movieNm,director,actor1,actor2,actor3,actor4,actor5,actor6,year,month,genre,showTm,grade,nation,company,starScore,userCount,screen,audience,show],
#           0x       1x     2o     3o    4o       5o    6o     7o    8o    9o    10o    11o   12o   13o     14o      15o      16o       17o    18o     19o
#                          배우이름--   2018, 01,  전처리전, 90/120, 전체/청불, 국가는 입력, 배급사도 입력,별점, 별점참여수, ..
# 바로 넣어줘도 되는것
# 별점, 별점참여수, 스크린수, 관람객 수, 상영횟수
# data = ["신","김","하정우","주","김","마","김","조","209","01","사극","90분 이하","전체관람가","한국","리얼",7.74,378,91,216,79]
# # data = []
# data = ["쥬라기 월드: 폴른 킹덤","후안 안토니오 바요나","크리스 프랫","브라이스 달라스 하워드","제프 골드블럼",
#         "저스티스 스미스","토비 존스","B.D. 웡","2018","06","액션","120분 이상 150분 이하","12세이상관람가",
#         "미국","유니버설픽쳐스인터내셔널 코리아(유)",8.1,15826,1,624,1]
# data = ["신과함께-죄와벌","김용화","하정우","차태현","주지훈","김향기","마동석","김동욱","2017","12","판타지","120분 이상 150분 이하",
#         "12세이상관람가","한국","롯데쇼핑(주)롯데엔터테인먼트",7.87,57376,58,22500,62]
# data = ["신","김","오달수","유해진","하정우","이경영","황정민","김윤석","2018","07","사극","150분 이상","12세이상관람가","한국","디즈니",10,671325,5000,10000,300000]
# data = ["레버넌트: 죽음에서 돌아온 자","알레한드로 곤잘레스 이냐리투","레오나르도 디카프리오","톰 하디","","","","",
#          "2015","01","어드벤처","150분 이상","15세이상관람가","미국","이십세기폭스코리아(주)","7.79","9676","9","2566","9"]
# data = ["마녀","박훈정","김다미","조민수","박희순","최우식","백승철","김하나","2018","06","미스터리","120분 이상 150분 이하",
#         "15세이상관람가","한국","워너브러더스 코리아(주)","8.2","20447","28","7836","32"]
data = []
initial_data=["","","","","","","","","2018","","","","","한국","",0,0,0,0,0]
for i in range(20):
    data.append(unquote(sys.argv[i+1]))
    if unquote(sys.argv[i+1])=="":
        data[i]=initial_data[i]

#2~7 배우
for j in range(2,8):
    for i in range(0, len(actor_score_set)): # 2018년 이후는 2018로 예외처리해야함
        if(data[j] == str(actor_score_set["actor"][i])):
            data[j] = round(actor_score_set["actor_score"][i],1)
            break
for i in range(2,8):
    if type(data[i])==str:
        data[i] = 2.0

#8 제작연도
for i in range(0, len(prdtYear_score_set)): # 2018년 이후는 2018로 예외처리해야함
    if(data[8] == str(prdtYear_score_set["prdtYear"][i])):
        data[8] = round(prdtYear_score_set["prdtYear_score"][i],1)
        break
    elif(int(data[8])>2018):
        data[8] = round(prdtYear_score_set["prdtYear_score"][16],1)
        break
    elif(int(data[8])<2002):
        data[8] = round(prdtYear_score_set["prdtYear_score"][0],1)
        break

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
#9 개봉월
for i in range(0,len(openDt_score_set)):
    if(month_change(data[9]) == openDt_score_set["openDt"][i]):
        data[9] = round(openDt_score_set["openDt_score"][i],1)
        break;
#10 장르
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
for i in range(0,len(genre_score_set)):
    if(genre_change(data[10]) == genre_score_set["repGenreNm"][i]):
        data[10] = round(genre_score_set["repGenreNm_score"][i],1)
        break;
#11 상영시간
def time_change(i):
    if i == "90분 이하":
        return 'under_90' # 90분 이하
    elif i == "90분 ~ 120분": #90분 ~ 120분
        return '90_120'
    elif i == "120분 ~ 150분": #120분 ~ 150분
        return '120_150'
    else:
        return '150_up' #150분 이상
for i in range(0,len(showTm_score_set)):
    if(time_change(data[11]) == showTm_score_set["showTm"][i]):
        data[11] = round(showTm_score_set["showTm_score"][i],1)
        break;
#12 등급
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
for i in range(0,len(watchGrade_score_set)):
    if(grade_change(data[12]) == watchGrade_score_set["watchGradeNm"][i]):
        data[12] = round(watchGrade_score_set["watchGradeNm_score"][i],1)
        break;
#13 국가
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
    elif i == '한국':
        return 'korea'
    elif i == '인도':
        return 'India'
    elif i == '미국':
        return 'america'
    elif i == '일본':
        return 'japan'
    else:
        return 'other_nation'
for i in range(0,len(nation_score_set)):
    if(nation_change(data[13]) == nation_score_set["repNationNm"][i]):
        data[13] = round(nation_score_set["repNationNm_score"][i],2)
        break
# 14 배급사
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
for i in range(0,len(distributor_score_set)):
    if(company_change(data[14]) == distributor_score_set["distributor"][i]):
        data[14] = round(distributor_score_set["distributor_score"][i],1)
        break

actor_sum = 0
for i in range(2,8):
    actor_sum += data[i] #actor

kk.loc[(len(kk))]= [
    data[9],   # openDt_score          float64
    data[8],   # prdtYear_score        float64
    data[13],   # repNationNm_score     float64
    data[10],   # repGenreNm_score      float64
    data[11],   # showTm_score          float64
    data[12],   # watchGradeNm_score    float64
    data[14],   # distributor_score     float64
    actor_sum,   # actor_score             int64
    float(data[15]),   # starScore             float64
    int(data[16]),   # userCount               int64
    int(data[17]),  # previous_screen         int64
    int(data[18]),  # previous_show           int64
    int(data[19])   # previous_audience       int64
]
# print(kk)

# print(openDt_score_set)     # float
# print(prdtYear_score_set)   #2018년 이후는 2018년 점수로 float
# print(nation_score_set)     # 얘 float
# print(genre_score_set)      # 얘 float
# print(showTm_score_set)     # 얘
# print(watchGrade_score_set) # 얘
# print(director_score_set)
# print(actor_score_set)
# print(distributor_score_set) #  얘
# print(test_set)
#콤보박스로 선택한 값이 배열형태로 오면
# 데이터프레임안에 넣고나서

# print(kk)
# print("="*100)
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

Ypredict = pickle_model.predict(kk)
print(int(Ypredict))
# 가져온거를 바꿔서 예측했고 그걸 내보내기만 하면 됨

# if __name__ == '__main__':
#     main()