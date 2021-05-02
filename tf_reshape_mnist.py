import tensorflow as tf
import pandas as pd

(ind_var,dep_var),_ = tf.keras.datasets.mnist.load_data()
ind_var = ind_var.reshape(784)
dep_var = pd.get_dummies(dep_var)
print(ind_var.shape,dep_var.shape)
#모델의 독립변수와 종속변수를 설정한다.

x = tf.keras.layers.Input(shape =[784])
h = tf.keras.layers.Dense(84, activation = "swish")(x)
y = tf.keras.layers.Dense(10, activation = "softmax")(h)
model = tf.keras.models.Model(x,y)
model.compile(loss = "categorical_crossentropy", metrics = "accuracy")
#모델을 생성한다.

model.fit(ind_var, dep_var, epochs=10)
#모델을 10번 학습시킨다.

print("Predictions : ", model.predict(ind_var[0:5]))
print("result : ", dep_var[0:5])
#모델을 이용하여 나온 결과값과 원래 결과값을 비교한다.
