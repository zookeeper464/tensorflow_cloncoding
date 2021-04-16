#모듈 입력모듈 입력한다.
import pandas as pd
import tensorflow as tf

# 0. 파일들로부터 데이터를 읽어온다.
file_path = "csv/lemonade.csv" #변수가 위치한 주소 입력
lemonade = pd.read_csv(file_path)

# 표 데이터의 구조를 나타낸다.
#print(lemonade.shape)
# 칼럼 이름을 출력한다.
#print(lemonade.columns)
# 맨 위의 데이터 5개를 표로 출력한다.
#lemonade.head()

# 1. 과거의 데이터를 준비한다.
in_var = lemonade[["온도"]]
de_var = lemonade[["판매량"]]
# 두 변수는 표에서 각 데이터들을 표로 추출한 형태이다.
# print(in_var.shape, de_var.shape) 

# 2. 모델의 구조를 만든다.
X = tf.keras.layers.Input(shape = [1])
# 독립변수의 구조(shape = ???)를 지정한다.
Y = tf.keras.layers.Dense(1)(X)
# 종속변수가 받는 독립변수()(???)와 종속변수의 구조(???)()를 지정한다.
model = tf.keras.models.Model(X,Y)
# 독립변수와 종속변수를 통해 학습할 모델을 만든다.
model.compile(loss="mse")
# ????

# 3. 데이터로 모델을 학습한다.
model.fit(in_var,de_var, epochs=1000)#, verbose=0이라하면 출력 하지 않는다는 설정
#모델을 통해 학습할 횟수(epochs=???)를 지정한다.

# 4. 모델을 이용한다.
print(model.predict([[15]]))
#X의 15번째 데이터를 가지고 나온 결과 값을 출력한다.
