import numpy as np
import sys
import time
from keras.datasets import mnist

np.random.seed(1)


# Выборо подобласти в пакете
def get_image_section(layer, row_from, row_to, col_from, col_to):
    sub_section = layer[:, row_from: row_to, col_from: col_to]
    return sub_section.reshape(-1, 1, row_to - row_from, col_to - col_from)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

data_start_pos = 0
data_end_pos = 1000
image_width = 28
image_height = 28
number_of_output_classes = 10

images, labels = (x_train[data_start_pos:data_end_pos].reshape(data_end_pos, image_width*image_height) / 255,
                  y_train[data_start_pos:data_end_pos])

one_hot_labels = np.zeros((len(labels), number_of_output_classes))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), image_width*image_height) / 255
test_labels = np.zeros((len(y_test), number_of_output_classes))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1


def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (output ** 2)


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


alpha = 2
iterations = 300
pixels_per_image = 784
num_labels = 10
batch_size = 128

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01

weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        layer_0 = images[batch_start: batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        # layer_0 - пакет изображений 28 х 28 пикселей

        sects = list()
        # Цикл for последовательно выбирает подобласти (kernel_rows x kernel_cols) в изображениях
        for row_start in range(layer_0.shape[1] - kernel_cols):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer=layer_0,
                                         row_from=row_start,
                                         row_to=row_start + kernel_rows,
                                         col_from=col_start,
                                         col_to=col_start + kernel_cols)
                sects.append(sect)
                # 8 изображений в пакете -> 100 подобластей в каждом пакете -> 800 изображений меньшего размера
        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        # **    Передача их в прямом направлении через линейный слой с одним выходным нейроном ->           **
        # **        -> получение прогноза из этого линейного слоя для каждой подобласти в каждом пакете.    **

        # Если передать изображение в линейный слой с n выходными нейронами, на выходе получим тот же результат,
        #   что и при использовании n линейных слоев (ядер) в каждой позиции в изображении. (Используем этот вариант)

        # Выход сверточного слоя - серия двумерных изображений (выход каждого ядра в каждой позиции входных изображений)
        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in range(batch_size):
            labelset = labels[batch_start + k: batch_start + k + 1]
            _inc = int(np.argmax(layer_2[k: k + 1]) == np.argmax(labelset))
            correct_cnt += _inc
        layer_2_delta = (labels[batch_start: batch_end] - layer_2) / (batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(l1d_reshape)
        kernels -= alpha * k_update

    test_correct_cnt = 0

    for i in range(len(test_images)):
        layer_0 = test_images[i: i + 1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

        sects = list()
        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer=layer_0,
                                         row_from=row_start,
                                         row_to=row_start + kernel_rows,
                                         col_from=col_start,
                                         col_to=col_start + kernel_cols)
                sects.append(sect)
        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, weights_1_2)

        test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i: i + 1]))

    if j % 1 == 0:
        sys.stdout.write(f'\nI: {str(j)}  '
                         f'Test-Acc: {str(test_correct_cnt / float(len(test_images)))}  '
                         f'Train-Acc: {str(correct_cnt / float(len(images)))}')

# page 225

