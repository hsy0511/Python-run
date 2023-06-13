fish_data = np.column_stack((fish_length, fish_weight))

fish_target = np.concatenate((np.ones(35), np.zeros(14)))

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify = fish_target, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn = kn.score(test_input, test_target)

print(kn.predict([[25,150]]))

distances, indexes = kn.kneighbors([[25,150]])
# 거리       인덱스

plt.scatter(train_input[:,0], train_input[:,1])
# 모든 행의 지정 열을 인덱싱 한다

plt.scatter(25,150, marker = '^')
# 25cm 150g 물고기는 ▲로 표시한다.

plt.scatter(train_input[indexes,0], train_input[indexes, 1], merker = 'D')
# 배열 인덱싱을하여 배열 데이터의 첫번째 열인 길이와 두번째 열인 무게를 반환한다.
# 그리고 이웃한 가장 가까운 5개의 데이터를 ◆로 나타낸다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()

plt.scatter(train_input[:,0], train_input[:,1])
# 모든 행의 지정 열을 인덱싱 한다

plt.scatter(25,150,marker = '^')
# 25cm 150g 물고기는 ▲로 표시한다.

plt.scatter(train_input[indexes,0], train_input[indexes, 1], merker = 'D')
# 배열 인덱싱을하여 배열 데이터의 첫번째 열인 길이와 두번째 열인 무게를 반환한다.
# 그리고 이웃한 가장 가까운 5개의 데이터를 ◆로 나타낸다.

plt.xlim((0,1000))
# 축의 범위를 수동으로 지정한다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()

mean = np.mean(train_input, axis = 0)
# mean 함수를 이용하여 train_input 배열의 두개에 데이터의 평균을 구한다.

std = np.std(train_input,axis=0)
# std 함수를 이용하여 train_input 배열의 두개에 데이터의 표준편차를 구한다.
print(mean,std)

train_scaled = (train_input - mean) / std
print(train_scaled)

new = ([25,150] - mean) / std
# 수상한 도미 데이터도 표준점수로 변환시킨다.

plt.scatter(train_scaled[:,0], train_scaled[:,1])
# 모든 행의 지정 열을 인덱싱 한다

plt.scatter(new[0], new[1], marker= '^')
# 수상한 도미 데이터는 ▲로 표시한다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

kn.fit(train_scaled, train_target)
# 두개의 데이터를 훈련시킨다.

test_scaled = (test_input - mean) / std
# test_scaled 데이터를 표준점수로 지정한다.

kn.score(test_scaled, test_target)
# 훈련시킨 데이터를 테스트한다.

print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])
# 거리       인덱스

plt.scatter(train_scaled[:,0], train_scaled[:,1])
# 모든 행의 지정 열을 인덱싱 한다

plt.scatter(new[0], new[1], marker='^')
# 수상한 도미 는 ▲로 표시한다.

plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
# 배열 인덱싱을하여 배열 데이터의 첫번째 열인 길이와 두번째 열인 무게를 반환한다.
# 그리고 이웃한 가장 가까운 5개의 데이터를 ◆로 나타낸다.

plt.xlabel('length')
plt.ylabel('weight')
plt.show()