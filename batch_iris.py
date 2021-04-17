import pandas as pd
import tensorflow as tf

# 0. 파일들로부터 데이터를 읽어온다.
file_path = "csv/iris.csv" #변수가 위치한 주소 입력
iris = pd.read_csv(file_path)

# 0-1. 원핫인코딩
iris = pd.get_dummies(iris)

# 1. 과거의 데이터를 준비한다.
in_var = iris[["꽃잎길이","꽃잎폭","꽃받침길이","꽃받침폭"]]
de_var = iris[["품종_setosa","품종_versicolor","품종_virginica"]]
# 두 변수는 표에서 각 데이터들을 표로 추출한 형태이다.
# print(in_var.shape, de_var.shape) 

# 2. 모델의 구조를 만든다.
X = tf.keras.layers.Input(shape = [4])
H = tf.keras.layers.Dense(8)(X)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation("swish")(H)

H = tf.keras.layers.Dense(6)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation("swish")(H)

H = tf.keras.layers.Dense(5)(H)
H = tf.keras.layers.BatchNormalization()(H)
H = tf.keras.layers.Activation("swish")(H)

Y = tf.keras.layers.Dense(3, activation = "softmax")(H)

model = tf.keras.models.Model(X,Y)
# 독립변수와 종속변수를 통해 학습할 모델을 만든다.
model.compile(loss="categorical_crossentropy", metrics = "accuracy")

# 3. 데이터로 모델을 학습한다.
model.fit(in_var,de_var, epochs=1000)#, verbose=0이라하면 출력 하지 않는다는 설정
#모델을 통해 학습할 횟수(epochs=???)를 지정한다.

# 4. 모델을 이용한다.
print(model.predict([[15]]))
#X의 15번째 데이터를 가지고 나온 결과 값을 출력한다.
