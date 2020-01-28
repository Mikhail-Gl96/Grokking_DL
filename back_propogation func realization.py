import numpy as np

np.random.seed(1)


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
        # print(f'weights[{index}] -= learning_rate ({learning_rate}) * derivative[index] ({derivative[index]})')
        weights[index] -= learning_rate * derivative[index]
        # weights[index] -= learning_rate * derivative[index]
    return weights


def relu(x):
    return (x > 0) * x


def relu2deriv(out):
    return out > 0


def nn(input, weight, goal_prediction, alpha=1.0, to_visual=0):
    print_star_num = 80

    layer_0 = input
    if to_visual == 1:
        print(f'layer_0 [{i}:{i + 1}] = {arr_print(layer_0, 10)}')
        print("*" * print_star_num)
    layer_1 = relu(get_prediction(layer=layer_0, weight=weight[0]))     # weight = weights_0_to_1
    if to_visual == 1:
        print(f'layer_1  = {arr_print(layer_1, 10)}')
        print(f'--> relu(np.dot(layer_0, weights_0_to_1))')
        print(f'  --> np.dot(layer_0, weights_0_to_1)): \n'
              f'      layer_0 = \n{arr_print(layer_0, 10)}\n'
              f'      weights_0_to_1 = {arr_print(weight[0], 10)}')
        print("*" * print_star_num)
    layer_2 = get_prediction(layer=layer_1, weight=weight[1])           # weight = weights_1_to_2
    if to_visual == 1:
        print(f'layer_2  = {arr_print(layer_2, 10)}')
        print(f'  --> np.dot(layer_1, weights_1_to_2)): \n'
              f'      layer_1 = {arr_print(layer_1, 10)}\n'
              f'      weights_1_to_2 = {arr_print(weight[1], 10)}')
        print("*" * print_star_num)

    # goal_prediction = walk_vs_stop[i:i + 1]
    layer_2_delta = get_delta(prediction=layer_2, goal_prediction=goal_prediction)
    if to_visual == 1:
        print(f'layer_2_delta  = {arr_print(layer_2_delta, 10)}')
        print("*" * print_star_num)
    layer_2_derivative = get_derivative(delta=layer_2_delta, layer=layer_1.T)
    error_2_layer = get_error(delta=layer_2_delta)
    if to_visual == 1:
        print(f'error_2_layer  = {error_2_layer}')
        print("*" * print_star_num)

    layer_1_delta = get_derivative(layer=layer_2_delta, delta=weight[1].T) * relu2deriv(layer_1)
    if to_visual == 1:
        print(f'layer_2_delta  = {arr_print(layer_1_delta, 10)}')
        print("*" * print_star_num)
    layer_1_derivative = get_derivative(delta=layer_1_delta, layer=layer_0.T)

    # weight = weights_0_to_1, weights_1_to_2
    derivative = layer_1_derivative, layer_2_derivative
    weight = get_weight(weights=weight, derivative=derivative, learning_rate=alpha)

    prediction = layer_2
    delta = layer_1_delta, layer_2_delta
    error = error_2_layer
    weights = weight
    return prediction, delta, error, derivative, weights


def arr_print(arr, symb_nth=0):
    arr_len = list(arr.shape)
    arr_out = []
    for dim in range(arr_len[0]):
        arr_current_string = []
        for dim_inside in arr[dim]:
            arr_current_string.append(dim_inside)
        arr_out.append(arr_current_string.copy())
    new = '\n'
    max_numbers_per_column = 8
    round_number = 5
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


streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([1, 1, 0, 0]).T

alpha = 0.2
hidden_size = 4

weights_0_to_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_to_2 = 2 * np.random.random((hidden_size, 1)) - 1
print(f'size weights_0_to_1 = {weights_0_to_1.shape}  \n'
      f'{arr_print(weights_0_to_1)}')
print(f'size weights_1_to_2 = {weights_1_to_2.shape}  \n'
      f'{arr_print(weights_1_to_2)}')

input = streetlights[0]  # 1, 0, 0
goal_prediction = walk_vs_stop[0]  # 0


weight = [weights_0_to_1, weights_1_to_2]


to_visual = 0
to_visual_part = 0
iter_to_show_detailed = [0, 1]


for ii in range(60):
    error_global = 0
    if (to_visual_part == 1) and (ii in iter_to_show_detailed):
        to_visual = 1
    for i in range(len(streetlights)):
        input = streetlights[i:i+1]            # change inputs
        goal_prediction = walk_vs_stop[i:i+1]  # change goal prediction
        # prediction, delta, error, derivative, weights
        results_my = nn(input=input, weight=weight, goal_prediction=goal_prediction, alpha=alpha, to_visual=to_visual)
        error_global += np.sum(results_my[2])       # compute error for each iteraration during 1 epoch
        weight = results_my[4]              # update weights
    if to_visual == 1:
        print("'"*150)
    if (to_visual_part == 1) and (ii in iter_to_show_detailed):
        to_visual = 0
    print(f'ITER = {ii}   --->   error_global = {error_global}')

# page 138
