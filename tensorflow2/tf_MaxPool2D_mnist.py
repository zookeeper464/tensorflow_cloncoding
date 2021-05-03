import tensorflow as tf
import pandas as pd

(ind_var,dep_var),_ = tf.keras.datasets.mnist.load_data()
ind_var = ind_var.reshape(60000,28,28,1)
dep_var = pd.get_dummies(dep_var)
print(ind_var.shape,dep_var.shape)

x = tf.keras.layers.Input(shape =[28,28,1])

h = tf.keras.layers.Conv2D(3, kernel_size=5, activation="swish")(x)
h = tf.keras.layers.MaxPool2D()(h)
#앞서 구성되었던 shape이 (24,24,3)에서 (12,12,3)으로 바뀐다.
#MaxPool2D는 2x2행렬에서 가장 큰 수를 고르는 방식으로 데이터를 처리하는 기법이다.
#Conv2D를 통해 유의미한 내용은 큰 수로 데이터가 남고 그 중에서 가장 큰 수만 남겨 데이터의 크기를 줄인다.

h = tf.keras.layers.Conv2D(6, kernel_size=5, activation="swish")(h)
h = tf.keras.layers.MaxPool2D()(h)
#앞서 구서오디었던 shape이 (8,8,6)에서 (4,4,6)으로 바뀐다.

h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(84, activation = "swish")(h)
y = tf.keras.layers.Dense(10, activation = "softmax")(h)
model = tf.keras.models.Model(x,y)
model.compile(loss = "categorical_crossentropy", metrics = "accuracy")

model.fit(ind_var, dep_var, epochs=10)

print(pd.DataFrame(model.predict(ind_var[0:5])).round(1))
