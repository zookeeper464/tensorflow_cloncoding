import tensorflow as tf

(mnist_x,mnist_y),_ = tf.keras.datasets.mnist.load_data()
print(mnist_x.shape, mnist_y.shape)
#tensorflow에 내장되어 있는 mnist 자료를 불러온다.

(cifar_x,cifar_y),_ = tf.keras.datasets.cifar10.load_data()
print(cifar_x.shape, cifar_y.shape)
#tensorflow에 내장되어 있는 cifar10 자료를 불러온다.

import matplotlib.pyplot as plt
plt.imshow(mnist_x[1], cmap = "gray")
#요소가 어떻게 이뤄져 있는지 관찰하게 해주는 모듈
#cmap은 흑백과 칼라를 구분하게 해준다.
#gray가 없어도 출력이 가능하지만 임의의 색으로 표현된다.
#칼라의 경우 gray를 넣어도 칼라로 표시된다.
print(mnist_y[0:10], end=" ")
#plt.imshow모듈은 항상 마지막에 표시된다.
#plt.imshow모듈은 입력된 마지막 값만 표시된다.

import numpy as np
#numpy.array는 리스트mport numpy as np
#numpy.array는 리스트의 차원을 보여준다.
d1 = np.array([1,2,3,4,5])
print(d1.shape)
d2 = np.array([d1,d1,d1,d1])
print(d2.shape)
d3 = np.array([d2,d2,d2])
print(d3.shape)
d4 = np.array([d3,d3])
print(d4.shape)

print(mnist_y.shape)
print(cifar_y.shape)
#두 리스트의 자료구조가 다름을 나타내준다.
#전자는 리스트이고, 후자는 길이가 1인 리스트들로 구성된 리스트의 형태로 나타난다.
