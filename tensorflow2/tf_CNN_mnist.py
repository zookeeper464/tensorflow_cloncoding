import tensorflow as tf
import pandas as pd

(ind_var,dep_var),_ = tf.keras.datasets.mnist.load_data()
ind_var = ind_var.reshape(60000,28,28,1)
dep_var = pd.get_dummies(dep_var)
print(ind_var.shape,dep_var.shape)

x = tf.keras.layers.Input(shape =[28,28,1])

h = tf.keras.layers.Conv2D(6, kernel_size=5, padding="same", activation="swish")(x)
#padding을 사용하지 않으면 픽셀의 크기가 줄어들지만 padding을 통해서 픽셀의 크기를 동일하게 유지할 수 있다.
h = tf.keras.layers.MaxPool2D()(h)
h = tf.keras.layers.Conv2D(16, kernel_size=5, activation="swish")(h)
h = tf.keras.layers.MaxPool2D()(h)

h = tf.keras.layers.Flatten()(h)
h = tf.keras.layers.Dense(120, activation = "swish")(h)
h = tf.keras.layers.Dense(84, activation = "swish")(h)
y = tf.keras.layers.Dense(10, activation = "softmax")(h)

model = tf.keras.models.Model(x,y)
model.compile(loss = "categorical_crossentropy", metrics = "accuracy")

model.fit(ind_var, dep_var, epochs=10)

print(pd.DataFrame(model.predict(ind_var[0:5])).round(1))
