import sys, numpy as np


from keras.datasets import mnist


def get_prediction(layer, weight):
    prediction_value = np.dot(layer, weight)
    return prediction_value


def get_delta(prediction, goal_prediction):
    delta_value = prediction - goal_prediction
    return delta_value


def get_error(delta):
    error_value = delta ** 2
    return error_value


def get_derivative(delta, layer):
    derivative_value = layer.dot(delta)
    return derivative_value


def get_weight(weights, derivative, learning_rate):
    for index in reversed(range(0, len(weights))):
        weights[index] -= learning_rate * derivative[index]
    return weights


def get_train_acc(train_acc, output, goal_prediction):
    return train_acc + int(np.argmax(output) == np.argmax(goal_prediction))


def relu(x):
    return (x >= 0) * x


def relu2deriv(out):
    return out >= 0


def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (output ** 2)


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def nn(input, weight, goal_prediction, alpha, train_acc=0, batch_size=100):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size

    _input = input[batch_start:batch_end]
    _goal_prediction = goal_prediction[batch_start:batch_end]

    layer_0 = _input
    layer_1 = tanh(get_prediction(layer=layer_0, weight=weight[0]))     # weight = weights_0_to_1
    dropout_mask = np.random.randint(2, size=layer_1.shape)
    layer_1 *= dropout_mask * 2
    layer_2 = softmax(get_prediction(layer=layer_1, weight=weight[1]))           # weight = weights_1_to_2

    layer_2_delta = get_delta(prediction=layer_2, goal_prediction=_goal_prediction)
    error_2_layer = get_error(delta=layer_2_delta)

    layer_1_delta = 0
    derivative = 0
    for k in range(batch_size):
        batch_size_goal_prediction = goal_prediction[batch_start + k:batch_start + k + 1]
        train_acc = get_train_acc(train_acc=train_acc, output=layer_2[k:k + 1], goal_prediction=batch_size_goal_prediction)

    # goal_prediction = walk_vs_stop[i:i + 1]
    layer_2_delta = get_delta(prediction=layer_2, goal_prediction=_goal_prediction) / (batch_size * layer_2.shape[0])
    layer_2_derivative = get_derivative(delta=layer_2_delta, layer=layer_1.T)

    layer_1_delta = get_derivative(layer=layer_2_delta, delta=weight[1].T) * tanh2deriv(layer_1)
    layer_1_delta *= dropout_mask
    layer_1_derivative = get_derivative(delta=layer_1_delta, layer=layer_0.T)

    derivative = layer_1_derivative, layer_2_derivative
    weight = get_weight(weights=weight, derivative=derivative, learning_rate=alpha)
    prediction = layer_2
    delta = layer_1_delta, layer_2_delta
    error = error_2_layer
    weights = weight
    return prediction, delta, error, derivative, weights, train_acc


def arr_print(arr, symb_nth=0):
    arr_len = list(arr.shape)
    arr_out = []
    for dim in range(arr_len[0]):
        arr_current_string = []
        for dim_inside in arr[dim]:
            arr_current_string.append(dim_inside)
        arr_out.append(arr_current_string.copy())
    new = '\n'
    round_number = 2
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

# 1000 numbers, 28 pix * 28 pix image size, 255 - colours ( we make it black and white )
images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])
print(f'images shape = {images.shape}  labels shape = {labels.shape}')

# create 10 classes of output numbers from 0 to 9
one_hot_labels = np.zeros((len(labels), 10))

# Label right answers to dataset's preset (i - pos of picture, l - class number)
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

# Create test set
test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
print(f'test_images shape = {test_images.shape}  test_labels shape = {test_labels.shape}')
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)

batch_size = 100
alpha = 2
iterations = 300
hidden_size = 100
pixels_per_image = 28 * 28  # 784
num_labels = 10

weights_0_1 = 0.02 * np.random.random((pixels_per_image, hidden_size)) - 0.01
weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

weight = [weights_0_1, weights_1_2]


for j in range(iterations):
    error = 0.0
    train_acc = 0
    test_acc = 0
    for i in range(int(len(images) / batch_size)):
        #      0        1      2        3          4         5
        # prediction, delta, error, derivative, weights, train_acc
        results_my = nn(input=images, weight=weight, goal_prediction=labels, alpha=alpha, train_acc=train_acc)

        error += np.sum(results_my[2])  # compute error for each iteraration during 1 epoch
        train_acc = results_my[5]       # Train acc
        weight = results_my[4]          # update weights

    sys.stdout.write(f'\r I: {j} Train-Error: {str(error / float(len(images)))[0:5]}  Train-Acc: {str(train_acc / float(len(images)))}')
    if (j % 10 == 0) or (j == iterations - 1):
        error_t, correct_cnt_t = (0.0, 0)
        for i in range(len(test_images)):
            layer_0_t = test_images[i:i + 1]
            layer_l_t = tanh(np.dot(layer_0_t, weights_0_1))
            layer_2_t = np.dot(layer_l_t, weights_1_2)
            error_t += np.sum((test_labels[i:i + 1] - layer_2_t) ** 2)
            correct_cnt_t += int(np.argmax(layer_2_t) == np.argmax(test_labels[i:i + 1]))
        sys.stdout.write("   Test-Err:" + str(error_t / float(len(test_images)))[0:5] + " Test-Acc:" + str(correct_cnt_t / float(len(test_images))))
        print()



# page 189

