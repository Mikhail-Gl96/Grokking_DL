import numpy as np


def get_prediction(input, weight):
    assert (len(input) == len(weight))
    prediction_value = 0
    for i in range(len(input)):
        prediction_value += (input[i] * weight[i])
        # print(f'{"".join(["*" for ii in range(i)])}*> prediction_value [{prediction_value}] '
        #       f'= prediction_value_prev [{prediction_value - input[i] * weight[i]}] + input[{i}] [{input[i]}] * weight[{i}] [{weight[i]}]')
    return prediction_value


def get_delta(prediction, goal_prediction):
    delta_value = prediction - goal_prediction
    # print(f'---> delta_value [{delta_value}] = prediction [{prediction}] - goal_prediction[{goal_prediction}]')
    return delta_value


def get_error(delta):
    error_value = delta**2
    # print(f'---> error_value [{error_value}] = delta [{delta}]**2')
    return error_value


def get_derivative(delta, input):
    derivative_value = np.array([0.0 for i in range(len(input))])
    # print(f'---> > > derivative_value = {derivative_value}')
    for i in range(len(input)):
        derivative_value[i] = delta * input[i]
        # print(f'{"".join(["*" for ii in range(i)])}*> derivative_value[{i}] [{derivative_value[i]}] '
        #       f'= delta [{delta}] * input[{i}] [{input[i]}]                   {delta * input[i]}')
    # print(f'---< < < < < < < derivative_value = {derivative_value}')
    return derivative_value


def get_weight(weights, derivative, learning_rate):
    weight_value = np.array([i for i in weights])
    # print(f'---> weight_value = {weight_value}')
    weight_value -= derivative * learning_rate
    weight_value[0] = 0  # FREEZE!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # print(f'--->>> weight_value [{weight_value}] = weight_value_prev [{weight_value + derivative * learning_rate} '
    #       f'- derivative [{derivative}] * learning_rate [{learning_rate}]')
    return weight_value


def nn(input, weight, goal_prediction, alpha=1.0):
    prediction = get_prediction(input=input, weight=weight)
    delta = get_delta(prediction=prediction, goal_prediction=goal_prediction)
    error = get_error(delta=delta)
    derivative = get_derivative(delta=delta, input=input)
    weights = get_weight(weights=weight, derivative=derivative, learning_rate=alpha)
    # print('\n\n')
    print(f'input = {input}\n'
          f'prediction = {prediction}\n'
          f'error = {error}\n'
          f'delta = {delta}\n'
          f'derivative(weight_deltas) = {derivative}\n'
          f'weights = {weights}')

    return prediction, delta, error, derivative, weights


weight = [0.1, 0.2, -0.1]

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]

input = [toes[0], wlrec[0], nfans[0]]
alpha = 0.3


# results_my = nn(input=input, weight=weight, goal_prediction=true, alpha=alpha)

for i in range(5):
    print(f'\n\nITER = {i}')
    # prediction, delta, error, derivative, weights
    results_my = nn(input=input, weight=weight, goal_prediction=true, alpha=alpha)
    weight = results_my[4]

