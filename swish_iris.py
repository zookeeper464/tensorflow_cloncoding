#모듈 입력모듈 입력한다.
import pandas as pd
import tensorflow as tf

# 0. 파일들로부터 데이터를 읽어온다.
file_path = "csv/iris.csv" #변수가 위치한 주소 입력
iris = pd.read_csv(file_path)

# 0-1. 원핫인코딩
iris = pd.get_dummies(iris) #회귀문제들과 다른 분류문제에서의 특징
# 종류를 숫자로 구성된 리스트로 바꾸어 저장시켜준다. 이를 원핫인코딩이라 한다.
#onehot-encoding
#column의 제목과 각 구성요소의 내용으로 새롭게 column의 제목이 생긴다. (제목)_(내용)

# 1. 과거의 데이터를 준비한다.
in_var = iris[["꽃잎길이","꽃잎폭","꽃받침길이","꽃받침폭"]]
de_var = iris[["품종_setosa","품종_versicolor","품종_virginica"]]
# 두 변수는 표에서 각 데이터들을 표로 추출한 형태이다.
# print(in_var.shape, de_var.shape) 

# 2. 모델의 구조를 만든다.
X = tf.keras.layers.Input(shape = [4])
# 독립변수의 구조(shape = ???)를 지정한다.
Y = tf.keras.layers.Dense(3, activation = "softmax")(X) # activation : 활성화 함수
#softmax는 종속변수를 확률로써 계산하게끔해준다. 만약 이 부분에 함수지정이 안되었다면 이는 identity라는 항등함수로 받아들인다.
# 종속변수가 받는 독립변수와 종속변수의 구조(종속변수의 종류)()를 지정한다.

model = tf.keras.models.Model(X,Y)
# 독립변수와 종속변수를 통해 학습할 모델을 만든다.
model.compile(loss="categorical_crossentropy", metrics = "accuracy")
# 회귀에 사용하는 loss는 mse이고 분류는 categorical_crossentropy이다.
# 메트릭스는 loss와 함께 보여주기 위한 값이고 그 값은 정확도이다.

# 3. 데이터로 모델을 학습한다.
model.fit(in_var,de_var, epochs=1000)#, verbose=0이라하면 출력 하지 않는다는 설정
#모델을 통해 학습할 횟수(epochs=???)를 지정한다.

# 4. 모델을 이용한다.
print(model.predict([[15]]))
#X의 15번째 데이터를 가지고 나온 결과 값을 출력한다.
