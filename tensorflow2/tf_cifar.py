import tensorflow as tf
import pandas as pd

(ind_var,dep_var),_ = tf.keras.datasets.cifar10.load_data()
dep_var = pd.get_dummies(dep_var.reshape(50000))
#dep_var은 2차원 구조로 이뤄져 있으므로 1차원으로 바꾸어 원핫인코딩을 진행한다.
print(ind_var.shape, dep_var.shape)
 

x = tf.keras.layers.Input(shape =[32,32,3])

h = tf.keras.layers.Conv2D(6, kernel_size=5, activation="swish")(x)
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
print(model.summary())
