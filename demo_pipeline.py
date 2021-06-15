import pickle
from pororo import Pororo
from transformers import AutoTokenizer,AutoModel
import tensorflow as tf
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity

zsl = Pororo(task="zero-topic", lang="ko")
tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart")
model_Bert = AutoModel.from_pretrained("hyunwoongko/kobart")

with open('/content/drive/MyDrive/MusicRecommendation/embeded_lyric.dat','rb') as fp:
    lyric = pickle.load(fp)

data = pd.read_csv('/content/drive/MyDrive/MusicRecommendation/music_data.csv')

diary = input('일기를 작성해주세요: \n')

add = input(f"{diary}\n까지 작성하셨습니다. 추가하실 내용이 없으세요? (Y/N)")
if add == 'Y':
    pass
elif add =='N':
    additional_diary = input('추가하실 내용을 적어 주세요: \n')
    diary += additional_diary

print('\n\n감정 분석을 진행하고 있습니다! 잠시만 기다려주세요 :)')

emotions = ['기쁨','신뢰','외로움','당황함','두려움','슬픔','걱정','분노']
result = zsl(diary,emotions)

filtered = [key for key,val in result.items() if val>70]

if len(filtered) == 0:
    sorted_result = sorted(result.items(), key=lambda kv: kv[1])[-3:]
    filtered = [key for key,val in sorted_result]

if len(filtered) == 1:
    selected = filtered[0]
    print(f'분석 결과 ##{selected}## 과 가장 유사한 감정을 느끼고 계신 것 같습니다.')
else: 
    print('아래 감정들 중 본인의 감정과 가장 유사한 것을 입력해주세요.')
    print(filtered)
    selected = input('위 감정들 중 하나를 선택해 주세요: \n')

actions = {
    '기쁨':['샤워','비눗방울','영화 감상','캠핑','산책','여행','피크닉','웃기','깜짝쇼','애완동물 놀아주기','요리하기','춤추기', '놀이동산 가기', '맥주 마시기', '게임하기','책상정리하기','와인마시기','노래하기','등산가기','맛있는 음식 먹기','로또 구입하기','운동하기', '소비하기'],
    '신뢰':['친구와 대화하기', '보드게임 하기','편지 쓰기','통화하기','선물하기','sns 하기', '친구와 약속잡기','여행 가기', '클래기 듣기', '봉사 활동 하기','운동하기','맛집 탐방하기','노래방 가기'],
    '외로움':['일기 쓰기', '원데이클래스 듣기','춤추기', '휴대폰 사진 정리', '추억 떠올리기', '독서하기', '사색하기','봉사활동 하기','동아리 활동하기','친구 만나기','자기 계발','뉴스 보기', '차 마시기','유투브 보기','편지 쓰기','여행가기', '음악 듣기', '종이접기', '스트레칭하기', '사람구경하기','술 마시기','운동하기','친구와 통화하기','화상채팅','쇼핑하기','운동하기','인스타하기','영화보기'],
    '당황함': ['심호흡하기','뜨개질하기', '일기 쓰기','반신욕하기','명상하기','와인 마시기','독서하기','산책하기','잠자기','웃어버리기','무시하기','차 마시기','화장실 다녀오기','잘못 인정하기','샤워하기','물 마시기','머리빗기','목욕하기','아로마 오일', '상황 파악하기'],
    '두려움': ['도움 요청하기', '귀여운 동물 검색하기', '친구에게 연락하기' ,'초콜릿 먹기', '소지품 챙기기',' 샤워하기','가족에게 연락하기', '종교 서적 읽기','크게 호흡하기','주변인과 의논하기','상황 파악하기'],
    '슬픔' : ['소리내어 울기', '꽃 구경하기', '이불 뒤집어 쓰고 자기',' 햇빛 쬐기','운동하기','억지로 무언가 하지 않기','주변인과 대화하기','샤워하기','청소하기','슬프게 하는 것들을 글로 적기'],
    '걱정' : ['스케줄러 쓰기','맛있는거 먹기','공부하기','상황 파악하기','매운 음식 먹기','디저트 먹기','여행가기','주변 사람과 걱정거리 나누기','다른 생각하기','와인 마시기','심호흡하기','술 마시기','서랍 정리하기','머리 스타일 바꾸기','방청소 하기'],
    '분노' : ['격한 운동하기','귀여운 동물 구경하기','술 마시기','춤추기','용서하기','영화보기','바람 쐬기','심호흡하기','왜 화났는지 써보기','상황 파악하기','코미디 보기','믿을 수 있는 사람에게 연락하기']
}

print(f'\n다음은 저희가 {selected} 감정을 느낄때 추천하는 것 들입니다. 관심이 가는 것들을 선택해주세요!')
print(f'{actions[selected]}')
selected_actions = input('마음에 드는 행동을 적어주세요: ')
print('\n어울리는 노래를 탐색중입니다. 잠시만 기다려주세요.')

tokenized_text = tokenizer.encode_plus(' '.join(selected_actions),padding=True, truncation=True,return_tensors='pt')['input_ids']
l = model_Bert(tokenized_text)['last_hidden_state']
l = l.detach().numpy()
pooled_l = tf.nn.max_pool1d(l,l.shape[1],l.shape[1],padding='SAME')

lf = [tf.squeeze(n) for n in lyric]
l = tf.reshape(pooled_l,[1,768])
cs_result = cosine_similarity(lf,l)

max = 0
max_n = 0
for n,r in enumerate(cs_result):
    if r > max:
        max=r
        max_n = n
time.sleep(3)
print(f"\n{data.iloc[max_n]['artist']}의 {data.iloc[max_n]['title']}을(를) 들으며 기분을 환기 해보는 걸 추천해드려요! \n사용해주셔서 감사해요!")
