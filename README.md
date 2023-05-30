# Python-run
# 인공지능, 머신러닝, 딥러닝
강의 : 혼자 공부하는 머신러닝+딥러닝 

url : https://www.youtube.com/playlist?list=PLVsNizTWUw7HpqmdphX9hgyWl15nobgQX
### 1,2,3강 내용 : https://colab.research.google.com/drive/1XqPM5tEmg1vGtsT5J-w7FZR_c4_GUA7F?usp=share_link
### 4강 내용 : 


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
1. 지도 학습 : 정답이 있는 데이터를 활용해 입력 값이 주어지면 입력값에 대한 Label를 주어 학습한다. (예 : k-최근접 이웃)
2. 비지도 학습 : 정답 라벨이 없는 데이터를 비슷한 특징끼리 군집화 하여 새로운 데이터에 대한 결과를 예측하여 학습한다. (상품 파악)
###### ※ 군집화 : 주어진 데이터 집합을 유사한 데이터들의 그룹으로 나누는 것
3. 강화 학습 : 데이터가 존재하는 것도 아니고 데이터가 있어도 정답이 따로 정해져 있지 않으며 자신이 한 행동에 대해 보상을 받으며 학습한다. (예 : 알파고)

### 샘플링 편향 (훈련 세트와 테스트 세트로 나눠 평가)
샘플링 편향이란? 

샘플이 크더라도 표본 추출 방법이 잘못되어 일반적이지 않은 대표성 없는 데이터가 포함되는 경우를 뜻한다.

###### ※ 대표성 : 어떤 조직이나 대표단 따위를 대표하는 성질이나 특성
- 잘못된 데이터

훈련 세트에는 앞에 35개를 훈련 시키고, 테스트 세트는 뒤에 15개 테스트 시킨다.

도미 35마리를 훈련시키고 빙어 15마리를 테스트 시키면 당연히 0% 학습을 한다.

- 올바른 데이터

훈련 세트에서도 도미와 빙어가 섞여 있어야 하고

테스트 세트에서도 도미와 빙어가 섞여 있어야 한다.
### 샘플링 편향 (잘못된 데이터)
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
### 샘플링 편향 (올바른 데이터)
- 넘파이 사용하기
```python
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/ddf514f9-2f95-44da-be1e-c4a469ec5e78)
- 데이터 섞기
```python
index = np.arange(49) 0~48까지 1씩 증가하는 정수 배열 만들어줌
np.random.shuffle(index) index 배열 랜덤으로 섞기

train_input = input_arr[index[:35]] 0~34 인덱스까지 배열 슬라이싱
train_target = target_arr[index[:35]] 

test_input = input_arr[index[35:]] 35~마지막 인덱스까지 배열 슬라이싱
test_target = target_arr[index[35:]] 
```
- 데이터 나누고 확인하기
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
```python
kn = kn.fit(train_input, train_target) 훈련

kn.score(test_input, test_target) 테스트
```
![image](https://github.com/hsy0511/Python-run/assets/104752580/8f598b7e-b02a-4e23-bc87-e74c976d4d28)
