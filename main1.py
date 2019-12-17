import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
from matplotlib import pyplot as plt

import numpy as np
from PIL import Image

# numpy.random.seed(42)
#
# train = [[], []]
# test = [[], []]
#
# # f = open("data.txt", "a")
# for i in range(21):
#     img = Image.open(str(i+1) + '_c_32.jpg')#.convert('L')
#     arr = np.asarray(img, dtype='uint8')#.tolist()
#     # for j in range(len(arr)):
#     #     for k in range(len(arr[j])):
#     #         arr[j][k] = [arr[j][k]]
#     arr = np.asarray(arr)
#     # arr = arr.tolist()
#     if i < 11:
#         if i != 10:
#             train[0].append(arr)
#             train[1].append([1])
#         else:
#             test[0].append(arr)
#             test[1].append([1])
#     else:
#         if i != 20:
#             train[0].append(arr)
#             train[1].append([0])
#         else:
#             test[0].append(arr)
#             test[1].append([0])
# # f.close()
#
# tr_0 = np.asarray(train[0])
# tr_1 = np.asarray(train[1])
# te_0 = np.asarray(test[0])
# te_1 = np.asarray(test[1])
#
#
# # Завантаження даних
# # (X_train, y_train), (X_test, y_test) = cifar10.load_data()#train, test
# X_train = np.asarray(tr_0)
# y_train = tr_1
# X_test = np.asarray(te_0)
# y_test = te_1
# # Розмір міні-вибірки
# batch_size = 32
# # Кількість класів зображень
# nb_classes = 10
# # Кількість епох навчання
# nb_epoch = 25
# # Розмір зображення
# img_rows, img_cols = 32, 32
# # Кількість каналів: RGB
# img_channels = 3
#
# # Нормалізація даних
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
#
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
#
# # Створення нейромережевої моделі
# model = Sequential()
# # Перший шар згортки
# model.add(Conv2D(32, (3, 3), padding='same',
#                         input_shape=(32, 32, 3), activation='relu'))
# # Друний шар згортки
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# # Перший шар субдискретизаії
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # Перший шар Dropout
# model.add(Dropout(0.25))
#
# # Третій шар згортки
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# # Четвертий шар згортки
# model.add(Conv2D(64, (3, 3), activation='relu'))
# # Другий шар субдисктеризації
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # Другий шар Dropout
# model.add(Dropout(0.25))
# # Шар перетворення вхідних даних
# model.add(Flatten())
# # Повнозв’язний шар
# model.add(Dense(512, activation='relu'))
# # Третій шар Dropout
# model.add(Dropout(0.5))
# # Вихідний шар
# model.add(Dense(nb_classes, activation='softmax'))
#
# # Параметри оптимізації
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
# # Навчання  моделі
# model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               epochs=nb_epoch,
#               validation_split=0.1,
#               shuffle=True,
#               verbose=2)
#
# # Оцінка якості навчання на тестових даних
# scores = model.evaluate(X_test, Y_test, verbose=0)
# print("Accuracy on test data: %.2f%%" % (scores[1]*100))
# # Збереження моделі
# model.save('my_model.h5')


def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print('The resized image size is {wide} wide x {height} '
          'high'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)


model = load_model('my_model_1.h5')
resize_image(input_image_path='24.jpeg',
                 output_image_path='24_32.jpg',
                 size=(32, 32))
resize_image(input_image_path='25.jpg',
                 output_image_path='25_32.jpg',
                 size=(32, 32))
for i in range(4, 6):
    img = Image.open('2'+str(i)+'_32.jpg')#.convert('L')
    x = np.asarray(img, dtype='uint8')#.tolist()
    x = np.asarray([x])
    re = model.predict(x)
    print(re)

    plt.imshow(re, interpolation='nearest')
    plt.show()




# на запас
# f = open("data.txt", "r")
# photo = f.read()
# photo = photo.split("\n")
# for i in range(len(photo)):
#     photo[i] = photo[i].split(";")
#     photo[i][0] = list(photo[i][0])
#
# print(photo)