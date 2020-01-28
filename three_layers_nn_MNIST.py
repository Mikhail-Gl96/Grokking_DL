import sys, numpy as np

from keras.datasets import mnist


def arr_print(arr, symb_nth=0):
    arr_len = list(arr.shape)
    arr_out = []
    for dim in range(arr_len[0]):
        arr_current_string = []
        for dim_inside in arr[dim]:
            arr_current_string.append(dim_inside)
        arr_out.append(arr_current_string.copy())
    new = '\n'
    round_number = 0
    max_numbers_per_column = 3 + round_number
    def gen_nth(length):
        return "".join([" " for i in range(length)])
    def check_enough_elements(string, length):
        add_number = 0
        if len(string) < length:
            add_number = length - len(string)
        return add_number
    line_horizontal = gen_nth(symb_nth) + "-" * (arr_len[1]*(max_numbers_per_column + 3) + 1)
    generated_data = f"{new}{gen_nth(symb_nth)}| ".join(["".join([str(iii) + gen_nth(check_enough_elements(str(iii), max_numbers_per_column)) + " | " for iii in np.around(i, decimals=round_number)]) for i in arr_out])
    return f'\n{line_horizontal}\n{gen_nth(symb_nth)}| {generated_data}\n{line_horizontal}'


# x_train - pics as array of numbers from 0 to 255; y_train - labels as a number represented in the pic
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f'x_train = {arr_print(x_train[11])}')

# 1000 numbers, 28 pix * 28 pix image size, 255 - colours ( we make black and white it)
images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])
print(f'images shape = {images.shape}  labels shape = {labels.shape}')

# create 10 classes of output numbers from 0 to 9
one_hot_labels = np.zeros((len(labels), 10))

# Label right answers to dataset's preset (i - pos of picture, l - class number)
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
    # print(f'i = {i}   l = {l}  one_hot_labels[{i}] = {one_hot_labels[i]}')
labels = one_hot_labels

# Create test set
test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
print(f'test_images shape = {test_images.shape}  test_labels shape = {test_labels.shape}')
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)

relu = lambda x: (x >= 0) * x
relu2div = lambda x: x >= 0

alpha = 0.005
iterations = 350
hidden_size = 40
pixels_per_image = 28 * 28  # 784
num_labels = 10

weights_0_1 = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error = 0.0
    correct_cnt= 0

    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2div(layer_1)

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
    sys.stdout.write(f'\r I: {j} Error: {str(error / float(len(images)))[0:5]}  Correct: {str(correct_cnt / float(len(images)))}')



