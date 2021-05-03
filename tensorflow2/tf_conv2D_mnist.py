import tensorflow as tf
import pandas as pd

(ind_var,dep_var),_ = tf.keras.datasets.mnist.load_data()
ind_var = ind_var.reshape(60000,28,28,1)
#tensorflow개발진들이 conv2D 사용시 3차원 shape으로 사용하라고 설정했다.
dep_var = pd.get_dummies(dep_var)
print(ind_var.shape,dep_var.shape)

x = tf.keras.layers.Input(shape =[28,28,1])
h = tf.keras.layers.Conv2D(3, kernel_size=5, activation="swish")(x)
h = tf.keras.layers.Conv2D(6, kernel_size=5, activation="swish")(h)
#독립변수를 받아서 특징맵 3개와 6개로 conv2D를 형성하고 이를 통해 표를 만든다.
#kernel_size는 특징맵의 크기를 뜻하며 특징맵은 5x5의 행렬로 구성되어있다.
#첫번째 : (5,5,1), 두번째 :(5,5,18)로 구성되어있다.
#이를 통해 구성된 내용은 (24,24,3), (20,20,6)이다.
h = tf.keras.layers.Flatten()(h)
#conv2D를 통해 만들어진 표를 Flatten을 통해 1차원 표로 바꾼다.
h = tf.keras.layers.Dense(84, activation = "swish")(h)
y = tf.keras.layers.Dense(10, activation = "softmax")(h)
model = tf.keras.models.Model(x,y)
model.compile(loss = "categorical_crossentropy", metrics = "accuracy")

model.fit(ind_var, dep_var, epochs=10)

print(pd.DataFrame(model.predict(ind_var[0:5])).round(1))
