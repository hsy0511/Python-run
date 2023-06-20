# Python-run
# 인공지능, 머신러닝, 딥러닝
강의 : 혼자 공부하는 머신러닝+딥러닝 

url : https://www.youtube.com/playlist?list=PLVsNizTWUw7HpqmdphX9hgyWl15nobgQX

강의 내용 : https://colab.research.google.com/drive/1XqPM5tEmg1vGtsT5J-w7FZR_c4_GUA7F?usp=share_link



## 제 1강 인공지능, 머신러닝 그리고 딥러닝이란 무엇인가?
인공지능 : 사람처럼 학습할 수 있고 사람 정도의 지능을 가지고 있는 지적 시스템

머신러닝 : 인공지능의 하위 분야 (인공지능의 소프트웨어)

머신러닝 라이브러리 : scikit learn

딥러닝 : 머신러닝의 하위분야 (인공 신경망 사용)

딥러닝 라이브러리 : tensorflow
## 제 2강 코랩과 주피터 노트북으로 손코딩 준비하기
구글에서 지원하는 무료 코딩 사이트 colab을 검색하여 홈페이지로 들어갑니다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/d6f06020-8641-4e94-b51a-a03baf3fe30b)

colab 홈페이지에서 새노트를 열어줍니다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/6902ecdd-bf04-49d3-863b-500616754135)

노트는 최대 5개 까지 열 수 있습니다.

노트를 열었으면 +코드와 +텍스트를 사용할 수 있는데 코드를 사용하려면 +코드를 누르면 되고 텍스트를 사용하려면 +텍스트를 사용하면된다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/c4340e6d-bc58-4e20-8f57-5c8c9e7a7a60)

코드 작성 박스에는 ▶ 표시가 있는데 이것은 실행 버튼이다.

그리고 colab을 사용할 때 다른 라이브러리를 설치하지 않아도 된다.

저장은 구글 드라이브에도 할 수 있어서 언제든지 볼 수 있다.

구글 드라이브에 저장된 코랩은 수시로 업데이트가 가능하여 새 노트를 만들지 않고 한노트에 계속해서 저장할 수 있다.

## 제 3강 마켓과 머신러닝
어느 마켓에 도미와 빙어를 비교하는 전통적인 프로그램
- 길이가 30이 넘으면 도미
- 비교할 두 클래스 도미와 빙어
- 데이터별 리스트로 분류
- 산점도 그리기
- 2차원 배열로 이진분류
### 도미 데이터
```python
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0,
                30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0,
                33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5,
                39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0,
                390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0,
                600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0,
                685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0,
                850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

print(bream_length)
print(bream_weight)

```

도미 데이터를 리스트로 표현했다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/75f10eac-2bdd-41c0-b64e-072a4b86ff34)
### 도미 데이터 산점도
```python
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

matplotib.pyplot 라이브러리를 불러와 그래프를 그릴 수 있게했다.

그래프에 형태는 scatter 함수를 이용하여 산점도 그래프로 만들었다.

show 함수로 그래프를 호출한다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/cf1eba32-4f68-4783-9778-2da121ed969b)
### 빙어 데이터
```python
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

print(smelt_length)
print(smelt_weight)
```

빙어 데이터도 리스트로 표현했다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/46111adf-ddc5-406b-8a98-778f044746ad)

### 도미, 빙어 데이터 산점도
```python
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

matplotib.pyplot 라이브러리를 불러와 그래프를 그릴 수 있게했다.

그래프에 형태는 scatter 함수를 이용하여 산점도 그래프로 만들었다.

show 함수로 그래프를 호출한다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/d3b5df2f-4bd5-4322-be82-f032dc317351)
### 도미, 빙어 데이터 길이와 무게로 분류
```python
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

print(length)
print(weight)
```

빙어의 길이와 도미의 길이를 length 변수안에 하나의 리스트로 연결해서 저장한다.

빙어의 무게와 도미의 무게를 weight 변수안에 하나의 리스트로 연결해서 저장한다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/7f7e77e3-a9f4-45d4-b700-60318f543a5f)
### 머신러닝 라이브러리 사이킷런이 원하는 데이터 형태 : 이진분류(2차원 배열)
![image](https://github.com/hsy0511/Python-run/assets/104752580/eaf93529-f018-4fb1-bd5e-c4c18971c93c)
### 이진분류
```python
fish_data = [[l,w] for l, w in zip(length, weight)]

print(fish_data)
```

zip 함수로 길이와 무게를 묶어서 길이는 l에 저장하고 무게는 w에 저장하여 리스트 안에 리스트들을 만들어 2차원배열을 생성한다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/96c79bf9-8e96-4c97-a23e-d00640faada6)
### 도미, 빙어 값 정하기
```python
fish_target = [1]*35 + [0]*14

print(fish_target)
```

리스트에 1이 35개 0이 14개인 리스트를 만든다.

여기서 우리는 1이 도미고 2가 빙어인것을 짐작하여 데이터를 정할 수 있다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/792a33d2-fae1-4b14-9b18-fe2d795843ea)
### k-최근접 이웃(머신러닝 알고리즘)으로 도미, 빙어 값이 무엇인지 확인하기
```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)
```

머신러닝의 라이브러리인 사이킷런에서 네이벌스 클래스객체를 불러온다.

네이벌스 클래스객체를 kn(모델)에 저장하고 fit 메서드을 통해서 fish_data와 fish_target에 머신러닝을 훈련 시켜줍니다.

잘 훈련되었는지 확인하기 위해 kn.score 메서드를 통해서 확인한다. 

score를 호출했을 때 값이 1.0이면 100% 정확도로 훈련한 것이다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/91f49573-3f66-4f5a-9b3d-80a1b0d5169f)
### 새로운 생선 예측
![image](https://github.com/hsy0511/Python-run/assets/104752580/40222ec6-03d1-4d1d-b293-32806bbec8a1)

```python
kn.predict([[30,600]])
```
길이가 30이고 무게가 600인 생선에 데이터를 predict 메서드로 예측할 수 있다.

predict의 값이 1이 나왔으면 도미인 것으로 예측한 것이고 그래프를 봐도 도미이기 때문에 이 생선은 도미인 것을 확인할 수 있다.

이것으로 머신러닝을 잘 훈련된 것도 알 수 있다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/b1631b66-9116-4367-a076-ba772b1245f1)
### 무조건 도미
```python
kn49 = KNeighborsClassifier(n_neighbors=49)

kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

print(35/49)
```

전체 샘플 49개를 다 바라보고 전체 샘플에 다수는 도미기 때문에 다 도미로 본다.

그리고 앞에서와 같이 fit으로 훈련시키고 score로 확인하면 71% 정도 학습한 것을 볼 수 있다.

49개중 35개가 도미니까 전체 샘플 49개중에 71%정도 도미인 것이기 때문에 잘 학습했다고 볼 수 있다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/481d52af-d8c5-4f7f-800d-6142d71d5d78)

## 제 4강 훈련 세트와 테스트 세트로 나누어 사용하기 
###  머신러닝 학습
![image](https://github.com/hsy0511/Python-run/assets/104752580/6e6e3868-ce39-4bf1-9d36-c9ce60dd6ef6)
1. 지도 학습 : 정답이 있는 데이터를 활용해 입력 값이 주어지면 입력값에 대한 Label를 주어 학습한다. (예 : k-최근접 이웃)

###### ※ label : 특징 
2. 비지도 학습 : 정답 라벨이 없는 데이터를 비슷한 특징끼리 군집화 하여 새로운 데이터에 대한 결과를 예측하여 학습한다. (상품 파악)
###### ※ 군집화 : 주어진 데이터 집합을 유사한 데이터들의 그룹으로 나누는 것
3. 강화 학습 : 데이터가 존재하는 것도 아니고 데이터가 있어도 정답이 따로 정해져 있지 않으며 자신이 한 행동에 대해 보상을 받으며 학습한다. (예 : 알파고)

### 샘플링 편향 (훈련 세트와 테스트 세트로 나눠 평가)
샘플링 편향이란? 

대표성 없는 학습 데이터이다.

즉, 샘플이 작으면 샘플링 잡음이 생기고, 표본 추출 방법이 잘못되면 대표성을 띄지 못할 수 있는 것을 말한다.
###### ※ 표본 추출 방법 : 전체 모집단을 몇 개의 하위 모집단으로 나눈 뒤, 각 하위 모집단들에서 독립적으로 표본을 추출하는 방법
###### ※ 모집단 : 정보를 얻고자 하는 관심 대상의 전체집합
###### ※ 샘플링 잡음 : 샘플 수가 작은 와중에 우연히 대표성이 없는 데이터가 포함되는 경우.
###### ※ 대표성 : 어떤 조직이나 대표단 따위를 대표하는 성질이나 특성

![image](https://github.com/hsy0511/Python-run/assets/104752580/3dcf7811-7f0e-4cea-8e0b-b6ce9b3bcf49)

- 잘못된 훈련 데이터

훈련 세트에는 앞에 35개를 훈련 시키고, 테스트 세트는 뒤에 15개 테스트 시킨다.

도미 35마리를 훈련시키고 빙어 15마리를 테스트 시키면 당연히 0% 학습을 한다.

- 올바른 훈련 데이터

훈련 세트에 도미와 빙어가 섞여 있고

테스트 세트에서도 도미와 빙어가 섞여 있기 때문에 100% 학습을 한다.
### 샘플링 편향 (잘못된 훈련 데이터)
앞에 데이터 도미 35마리를 훈련 시키고 뒤에 데이터 빙어 15마리를 테스트 시키기 때문에 당연히 0% 학습을 한다. 

1과 0은 도미와 빙어를 말한다. 1은 도미, 0은 빙어
```python
train_input = fish_data[:35] 0인덱스부터 34 인덱스까지 슬라이싱 
train_target = fish_target[:35]

test_input = fish_data[35:] 35인덱스부터 끝까지 슬라이싱
test_target = fish_target[35:]

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier() 클래스 객체 초기화
kn = kn.fit(train_input, train_target) 트레인 훈련

kn.score(test_input, test_target) 훈련 테스트
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/aa1e3a4b-6c1b-4301-9a00-0f0e493ae59d)

배열 데이터

![image](https://github.com/hsy0511/Python-run/assets/104752580/cd87de57-7643-44f4-9469-912bd9cf4741)

### 샘플링 편향 (올바른 훈련 데이터)

- 넘파이 사용하기

여기서 넘파이를 사용하는 이유는 리스트로만 표현하기에는 한계가 있을 수 있기 때문에 차원 배열로 표시하기 위해서 입니다.
```python
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/ddf514f9-2f95-44da-be1e-c4a469ec5e78)
- 데이터 섞기

넘파이 2차원 배열을 랜덤으로 섞어서 나타낸다.

1과 0은 도미와 빙어를 말한다. 1은 도미, 0은 빙어
###### ※ shuffle : 혼합 (리스트)
###### ※ ramdom.shuffle() : 리스트 섞기
```python
index = np.arange(49) 0~48까지 1씩 증가하는 정수 배열 만들어줌
np.random.shuffle(index) index 배열 랜덤으로 섞기

train_input = input_arr[index[:35]] 0~34 인덱스까지 배열 슬라이싱
train_target = target_arr[index[:35]] 

test_input = input_arr[index[35:]] 35~마지막 인덱스까지 배열 슬라이싱
test_target = target_arr[index[35:]] 
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/5f8f6030-2fe0-4834-bc31-f46a0fa8ca8b)


- 데이터 나누고 확인하기

그래프를 그려서 데이터가 섞였는지 확인한다.
```python
import matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:,1]) 첫번째 열에 전체 행을 선택한다. 두번째 열에 전체 행을 선택한다.
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/157c19ff-fb68-4633-b1d7-22fdb6303dd8)
- 결과 확인

섞여진 데이터가 학습이 되었는지 확인한다.
```python
kn = kn.fit(train_input, train_target) 훈련

kn.score(test_input, test_target) 테스트
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/8f598b7e-b02a-4e23-bc87-e74c976d4d28)

100% 학습이 완료 된것을 알 수 있다.

## 제 5강 정교한 결과 도출을 위한 데이터 전처리 알아보기
- 길이가 25cm이고 무게가 150g이 나가는 생선이 도미인지 빙어인지 알아보자

![image](https://github.com/hsy0511/Python-run/assets/104752580/96cdc631-34cb-481e-9ac1-4af31b9c502b)

- 넘파이를 이용하여 데이터를 준비한다.

```python
fish_data = np.column_stack((fish_length, fish_weight))
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/734ee2a8-37ce-4923-ad16-130541451938)

첫열은 length, 두번째 열은 weight이다.

column_stack을 사용하여 배열을 열방향으로 쌓는다.

- 35개 도미와 14개 빙어를 알기 위해서 도미는 1 빙어는 0으로 나타낸다.

```python
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/8577069a-1259-4bdc-9731-8ae739190dea)

np.ones()는 1로 채워주는 함수이고, np.zeros()는 0으로 채워주는 함수이다.

concatenate() 함수는 하나의 배열로 데이터를 묶어주는 함수이다.

즉, np.concatenate((np.ones(35), np.zeros(14)))는 1 데이터 35개와 0 데이터 14개가 같이 한 배열에 있는것을 의미한다.

- 사이킷런으로 데이터 나누기

```python
from sklearn.model_selection import train_test_split
```

사이킷런 모델 셀렉션 모듈 밑에 트레인 테스트 스플릿이라는 함수를 제공하고 있다.

훈련 세트와 테스트 세트로 나눈다.

```python
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify = fish_target, random_state = 42)
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/9394727a-e6fe-465a-bbc2-e810c51c1770)

train_test_split 함수는 입력 데이터와 타겟 데이터를 한꺼번에 전달할 수 있다.

stratify라는 매개변수는 특정 클래스의 샘플이 작을 때 사용한다.

stratify 매개변수를 random_state라는 난수 값과 같이 사용하여 fish_target 값이 섞이도록 해준다.

- 수상한 도미

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn = kn.score(test_input, test_target)
```

KNeighborsClassifier 모듈을 불러와서 주변 5개에 요소를 바라보면서 예측을 하고

kn 모델을 fit 메소드와 score 메소드로 훈련과 테스트를 시킨다.

![image](https://github.com/hsy0511/Python-run/assets/104752580/da9f6b30-1a57-4b9d-98fb-15f5e933fa46)

100% 훈련된 것 볼 수 있다.

이상한 도미 길이 25cm의 무게 150g을 predict 메소드로 클래스를 예측했다.

```python
print(kn.predict([[25,150]]))
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/3ed1a5f1-8dff-423f-b979-fd222f5fd09b)

0이 나왔다. 즉 빙어라는 예측이 나왔다.

좀 더 정확하게 확인하기 위해 kneighbors 메소드를 이용한다.

kneighbors 메소드는 k-최근접 이웃 알고리즘이 바라보는 이웃의 샘플을 뽑아낸다.

```python
distances, indexes = kn.kneighbors([[25,150]])
거리       인덱스

plt.scatter(train_input[:,0], train_input[:,1])
모든 행의 지정 열을 인덱싱 한다

plt.scatter(25,150, marker = '^')
25cm 150g 물고기는 ▲로 표시한다.

plt.scatter(train_input[indexes,0], train_input[indexes, 1], marker = 'D')
배열 인덱싱을하여 배열 데이터의 첫번째 열인 길이와 두번째 열인 무게를 반환한다.
그리고 이웃한 가장 가까운 5개의 데이터를 ◆로 나타낸다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/f4f78a9c-c1d4-400e-a108-6c28ce81f548)

그래프를 봤을 때 도미에 더 가까운것 같이 보이지만 무게 단위는 1000까지고 길이 단위는 40까지이기 때문에 실제로는 빙어와 같은것이 맞다.

- 기준을 맞춰라 

위에서 말한거와 같이 실제로 어느 물고기와 가까운 데이터인지 확인한다.

```python
plt.scatter(train_input[:,0], train_input[:,1])
모든 행의 지정 열을 인덱싱 한다

plt.scatter(25,150,marker = '^')
25cm 150g 물고기는 ▲로 표시한다.

plt.scatter(train_input[indexes,0], train_input[indexes, 1], marker = 'D')
배열 인덱싱을하여 배열 데이터의 첫번째 열인 길이와 두번째 열인 무게를 반환한다.
그리고 이웃한 가장 가까운 5개의 데이터를 ◆로 나타낸다.

plt.xlim((0,1000))
축의 범위를 수동으로 지정한다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/4b75a75c-b8d2-4f69-a0e1-8e7b4719b09a)

축의 범위를 1000으로 지정하고 봤을 때 수상한 생선과 가까운 5개의 생선은 빙어인 것을 더 정확하게 볼 수 있다.

하지만 여기서는 길이로는 물고기를 예측하지 않고 무게로만 예측했다.

- 표준 점수로 바꾸기 

![image](https://github.com/hsy0511/Python-run/assets/104752580/d7b4f62f-d969-466e-b65e-e614fb4ea992)

무게와 길이 두 특성을 사용하여 예측하는 것이다.

두 특성을 사용하여 예측하기 위해서는 무게아 길이의 평균과 표준편차를 구해야 합니다.

```python
mean = np.mean(train_input, axis = 0)
mean 함수를 이용하여 train_input 배열의 두개에 데이터의 평균을 구한다.

std = np.std(train_input,axis=0)
std 함수를 이용하여 train_input 배열의 두개에 데이터의 표준편차를 구한다.
print(mean,std)
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/9b1931a0-cd05-497a-a941-2434a49095de)

평균과 표준편차를 얻으면 표준 점수로 바꿔야하기 때문에 (특성 - 평균) / 표준 편차를 해야합니다.

```python
train_scaled = (train_input - mean) / std
print(train_scaled)
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/26d01a05-f643-4b75-a53f-e32ffdb75622)

train_scaled를 표준 점수로 지정해 줍니다.

- 수상한 도미 다시 표시하기

```python
new = ([25,150] - mean) / std
수상한 도미 데이터도 표준점수로 변환시킨다.

plt.scatter(train_scaled[:,0], train_scaled[:,1])
모든 행의 지정 열을 인덱싱 한다

plt.scatter(new[0], new[1], marker= '^')
수상한 도미 데이터는 ▲로 표시한다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/ddaaeb61-dddf-40ec-a884-345bac1707f2)

표준 점수의 그래프로 나타난다.

- 전처리 데이터에서 모델 훈련

표준 점수로 만든 그래프를 k-최근접 이웃으로 다시 그래프를 만든다.

```python
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(train_scaled, train_target)
두개의 데이터를 훈련시킨다.

test_scaled = (test_input - mean) / std
test_scaled 데이터를 표준점수로 지정한다.

kn.score(test_scaled, test_target)
훈련시킨 데이터를 테스트한다.
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/69fb2d22-6d0e-4093-8f65-8ec7d516c1e9)

100% 훈련된 것을 볼 수 있다.

```python
print(kn.predict([new]))
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/e1332aeb-3431-4da5-84c1-3d8723243744)

1로 예측했다. 즉 도미로 예측한 것이다.

```python
distances, indexes = kn.kneighbors([new])
거리       인덱스

plt.scatter(train_scaled[:,0], train_scaled[:,1])
모든 행의 지정 열을 인덱싱 한다

plt.scatter(new[0], new[1], marker='^')
수상한 도미 는 ▲로 표시한다.

plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
배열 인덱싱을하여 배열 데이터의 첫번째 열인 길이와 두번째 열인 무게를 반환한다.
그리고 이웃한 가장 가까운 5개의 데이터를 ◆로 나타낸다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/0b8cd057-8129-4b55-b75f-aaef5b30e69b)

표준 점수를 사용하여 k-최근접 이웃 그래프를 그렸을 때 가장 가까운 데이터 5개가 도미로 나온것을 보아 결국에 수상한 도미 데이터는 도미인 것을 알 수 있다.

## 제 6강, 회귀 문제를 이해하고 k-최근접 이웃 알고리즘으로 풀어 보기

- 농어의 무게를 예측하라

![image](https://github.com/hsy0511/Python-run/assets/104752580/c0629b00-cd99-4331-a3bf-7abc9c11ff90)

농어의 무게를 예측기위해 회귀를 사용한다.

길이를 사용해서 무게를 예측할 것이다.

- 회귀(regression)

회귀는 샘플의 특성값으로부터 다른 특성값을 유추하는 방법이다.

- k-최근접 이웃 회귀

![image](https://github.com/hsy0511/Python-run/assets/104752580/6ff0e0a8-933a-4bf5-a6dc-c3f60eea1f25)

이웃한 숫자의 평균값이 회귀에 예측 값이다.

- 농어의 길이만 사용

```python
import numpy as np
numpy 배열 사용

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

print(perch_length)

print(perch_weight)
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/8a1f4586-ffcf-44b9-a7f5-d4516707a530)

```python
import matplotlib.pyplot as plt

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
산점도 그래프를 그려서 농어 데이터를 쉽게 볼 수 있다.
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/f4b191ea-ea28-4a77-9fc5-f90f6dc0888b)

- 훈련 세트 준비

```python
from sklearn.model_selection import train_test_split
train_test_split 모듈 사용하여 배열을 나눠준다.

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state = 42)    
train_test_split 모듈 사용하여 훈련데이터와 테스트 배열을 1차원 배열로 나눠준다.

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
reshape(-1,1)에서 1은 하나의 열이 있는 2차원 배열을 생성한다는 것이다.
-1은 나머지 차원이 결정되고 남은 차원을 사용한다는 뜻이다.
  
print(train_input)
print(test_input)
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/5ab5e335-d331-4ea2-a6aa-93a80c3e9828)

트레인 데이터와 테스트 데이터를 2차원 배열로 바꿨다.

- 회귀 모델 훈련

```python
from sklearn.neighbors import KNeighborsRegressor
KNeighborsRegressor 모듈을 사용하여 k-최근접 이웃 회귀를 사용할 수 있다.

knr = KNeighborsRegressor()
모델에 클래스 객체 할당

knr.fit(train_input, train_target)
모델을 훈련시킨다.
  
knr.score(test_input, test_target)
모델을 테스트 시킨다.

```

![image](https://github.com/hsy0511/Python-run/assets/104752580/d2fb79e4-9333-421f-ad2d-50888fe3a7cf)

하지만 회귀는 분류와 달리 정확도가 나오는 것이 아니라 결정 계수가 나온다.

###### ※ 결정계수: 대상을 얼마나 잘 설명할 수 있는가를 숫자로 나타낸 것

### 결정계수(R2) 구하는 방법

![image](https://github.com/hsy0511/Python-run/assets/104752580/31bcfb7e-3ad1-414f-b5cd-fab3004515c4)

여기서 타겟은 무게이다.

R2가 1에 가까워지면 좋은 회귀 모델이다.

- 평균 구하기

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error라는 모듈을 사용하여 평균을 구한다.

test_prediction = knr.predict(test_input)
test_input의 예측한 값을 test_prediction에 저장한다.

mae = mean_absolute_error(test_target, test_prediction)
test_target과 test_prediction의 차이 평균을 절대 값으로 mae에 저장한다.
print(mae)
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/0833a2c8-9265-4ad7-8338-99d0715c0d8b)

- 과대적합과 과소적합

```python
knr.score(train_input, train_target)
훈련 세트를 테스트 시킨다.

knr.score(test_input, test_target)
테스트 세트를 테스트 시킨다.
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/315a3549-6334-4b6e-874e-a690f0576a02)

![image](https://github.com/hsy0511/Python-run/assets/104752580/d58ab05d-e639-42f5-a1d9-8ac069006e68)

원래 테스트 세트보다 훈련 세트가 더 테스트 값이 높게 나와야 되는데 여기서는 테스트 세트가 더 높게 나왔다.

이런 현상을 훈련 세트를 적절히 학습하지 못했다는 것으로 과소적합했다 라고 말한다.

또 반대로 너무 훈련에만 맞아서 테스트 세트가 실전에 투입했을 때 형판없는 모델이 된다는 것을 과대적합 이라고도 부릅니다.

- 이웃 개수 줄이기

k-최근접 이웃에서 k 개수를 늘리면 과소적합이 되고 극단적으로 k 개수를 줄이면 과대적합이 된다.

```python
knr.n_neighbors = 3
KNeighbors의 기본 값인 5가 아니라 3으로 줄여서 사용했다.

knr.fit(train_input, train_target)
knr 모델을 훈련시킨다.
  
print(knr.score(train_input, train_target))
훈련 세트 점수를 확인한다.

print(knr.score(test_input, test_target))
테스트 세트 점수를 확인한다.
```

![image](https://github.com/hsy0511/Python-run/assets/104752580/0e4ba808-64a4-47de-afb7-a0e45940ceed)

두 갮이 너무 동 떨어지지 않고 두 값이 같이 꽤 높은 값이 유지되면서 훈련 세트가 조금 더 높은 현상을 보여주며 과소적합과 과대적합에 균혀을 그래도 잘 맞췄다고 볼 수 있다.

- 그래프 확인

1. n_neighbors = 1 (이웃한 샘플 1개)

![image](https://github.com/hsy0511/Python-run/assets/104752580/ea839a80-985b-4cf5-adbf-ae8047c533cb)

들쭉 날쭉한 그래프 

2. n_neighbors = 3 (이웃한 샘플 3개)

![image](https://github.com/hsy0511/Python-run/assets/104752580/b5b68b48-1e5a-4fad-96d7-91a365af056c)

그래도 부드러운 곡선
3. n_neighbors = 42 (이웃한 샘플 42개)

![image](https://github.com/hsy0511/Python-run/assets/104752580/12922f7e-9383-4d88-b731-7de182ec0dcf)

하나의 값만 예측한다.

훈련 세트를 학습하지 못한 과소적합된 모델이다.
