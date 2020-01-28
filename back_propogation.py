import numpy as np

np.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2deriv(out):
    return out > 0


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
print(f'size weights_0_to_1 = {weights_0_to_1.shape}  \n'
      f'{arr_print(weights_0_to_1)}')

weights_1_to_2 = 2 * np.random.random((hidden_size, 1)) - 1
print(f'size weights_1_to_2 = {weights_1_to_2.shape}  \n'
      f'{arr_print(weights_1_to_2)}')

input = streetlights[0]  # 1, 0, 0
goal_prediction = walk_vs_stop[0]  # 0

to_visual = 0
to_visual_part = 0
iter_to_show_detailed = [0, 1]

for ii in range(60):
    if (to_visual_part == 1) and (ii in iter_to_show_detailed):
        to_visual = 1
    error_2_layer = 0
    print(f'\n\n\nITER = {ii}\n')
    print_star_num = 80
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        if to_visual == 1:
            print(f'layer_0 [{i}:{i + 1}] = {arr_print(layer_0, 10)}')
            print("*" * print_star_num)
        layer_1 = relu(np.dot(layer_0, weights_0_to_1))
        if to_visual == 1:
            print(f'layer_1  = {arr_print(layer_1, 10)}')
            print(f'--> relu(np.dot(layer_0, weights_0_to_1))')
            print(f'  --> np.dot(layer_0, weights_0_to_1)): \n'
                  f'      layer_0 = \n{arr_print(layer_0, 10)}\n'
                  f'      weights_0_to_1 = {arr_print(weights_0_to_1, 10)}')
            print("*" * print_star_num)
        layer_2 = np.dot(layer_1, weights_1_to_2)
        if to_visual == 1:
            print(f'layer_2  = {arr_print(layer_2, 10)}')
            print(f'  --> np.dot(layer_1, weights_1_to_2)): \n'
                  f'      layer_1 = {arr_print(layer_1, 10)}\n'
                  f'      weights_1_to_2 = {arr_print(weights_1_to_2, 10)}')
            print("*" * print_star_num)
        error_2_layer += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)
        if to_visual == 1:
            print(f'error_2_layer  = {error_2_layer}')
            print("*" * print_star_num)
        layer_2_delta = layer_2 - walk_vs_stop[i:i+1]
        if to_visual == 1:
            print(f'layer_2_delta  = {arr_print(layer_2_delta, 10)}')
            print("*" * print_star_num)
        layer_1_delta = layer_2_delta.dot(weights_1_to_2.T) * relu2deriv(layer_1)
        if to_visual == 1:
            print(f'layer_2_delta  = {arr_print(layer_1_delta, 10)}')
            print("*" * print_star_num)

        weights_1_to_2 -= alpha * layer_1.T.dot(layer_2_delta)
        weights_0_to_1 -= alpha * layer_0.T.dot(layer_1_delta)

    print(f'--->   error_2_layer = {error_2_layer}')
    if to_visual == 1:
        print("'"*150)
    if (to_visual_part == 1) and (ii in iter_to_show_detailed):
        to_visual = 0

# page 138

# import time
# time.sleep(20)
