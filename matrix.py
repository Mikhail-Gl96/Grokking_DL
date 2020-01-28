import numpy as np


def get_prediction(input, weight):
    prediction_value = input.dot(weight)
    return prediction_value


def get_delta(prediction, goal_prediction):
    delta_value = prediction - goal_prediction
    return delta_value


def get_error(delta):
    error_value = delta**2
    return error_value


def get_derivative(delta, input):
    derivative_value = delta * input
    return derivative_value


def get_weight(weights, derivative, learning_rate):
    weight_value = weights
    weight_value -= derivative * learning_rate
    return weight_value


def nn(input, weight, goal_prediction, alpha=1.0):
    prediction = get_prediction(input=input, weight=weight)
    delta = get_delta(prediction=prediction, goal_prediction=goal_prediction)
    error = get_error(delta=delta)
    derivative = get_derivative(delta=delta, input=input)
    weights = get_weight(weights=weight, derivative=derivative, learning_rate=alpha)
    # print('')
    # print(f'input = {input}\n'
    #       f'prediction = {prediction}\n'
    #       f'error = {error}\n'
    #       f'delta = {delta}\n'
    #       f'derivative(weight_deltas) = {derivative}\n'
    #       f'weights = {weights}')
    # print('- '*50)
    return prediction, delta, error, derivative, weights

#
# def test_nn(input, weight, goal_prediction):
#     prediction = get_prediction(input=input, weight=weight)
#     delta = get_delta(prediction=prediction, goal_prediction=goal_prediction)
#     error = get_error(delta=delta)
#     print(f'input = {input}\n'
#           f'prediction = {prediction}  goal prediction = {goal_prediction}\n'
#           f'error = {error}\n'
#           f'delta = {delta}\n')


streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 1],
                         [1, 0, 1]])

# walk_vs_stop = np.array([[0],
#                          [1],
#                          [0],
#                          [1],
#                          [1],
#                          [0]])
walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])

weghts = np.array([0.5, 0.48, -0.7])
alpha = 0.1

input = streetlights[0]  # 1, 0, 0
goal_prediction = walk_vs_stop[0]  # 0

for i in range(40):
    error_global = 0
    print(f'\n\nITER = {i}')
    for ii in range(len(walk_vs_stop)):
        input = streetlights[ii]            # change inputs
        goal_prediction = walk_vs_stop[ii]  # change goal prediction
        # prediction, delta, error, derivative, weights
        results_my = nn(input=input, weight=weghts, goal_prediction=goal_prediction, alpha=alpha)
        error_global += results_my[2]       # compute error for each iteraration during 1 epoch
        weight = results_my[4]              # update weights
    print(f'--->   error_global = {error_global}')

# page 138
