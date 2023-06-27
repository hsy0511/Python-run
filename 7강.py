print(knr.predict([[50]]))

distances, indexes = knr.kneighbors([[50]])

plt.scatter(train_input, train_target)

plt.scatter(train_input[indexes], train_target[indexes], marker='D')

plt.scatter(50, 1033, marker='^')
plt.show()

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train_input, train_target)

print(lr.predict([[50]]))

print(lr.coef_, lr.intercept_)

plt.scatter(train_input, train_target)


plt.plot([15,50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])


plt.scatter(50, 1241.8, marker='^')


plt.show()


print(lr.score(train_input, train_target))


print(lr.score(test_input, test_target))


train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

print(train_poly)
print(test_poly)

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50 ** 2, 50]]))

print(lr.coef_, lr.intercept_)

point = np.arange(15, 50)


plt.scatter(train_input, train_target)


plt.plot(point, 1.01 * point ** 2 -21.6 * point + 116.05)



plt.scatter([50], [1574], marker= '^')

plt.show()

print(lr.score(train_poly, train_target))


print(lr.score(test_poly, test_target))