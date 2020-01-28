import numpy as np

# 119 page


def get_prediction(input, weight):
    prediction_value = [0 for i in range(len(weight))]
    for i in range(len(weight)):
        prediction_value[i] += (input * weight[i])
    return prediction_value


def get_delta(prediction, goal_prediction):
    delta_value = np.array([prediction[i] - goal_prediction[i] for i in range(len(prediction))])
    return delta_value


def get_error(delta):
    error_value = np.array([delta[i]**2 for i in range(len(delta))])
    return error_value


def get_derivative(delta, input):
    derivative_value = np.array([delta[i] * input for i in range(len(delta))])
    return derivative_value


def get_weight(weights, derivative, learning_rate):
    weight_value = np.array([i for i in weights])
    for i in range(len(weight)):
        weight_value[i] -= derivative[i] * learning_rate
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


weight = [0.3, 0.2, 0.9]

wlrec = [0.65, 1.0, 1.0, 0.9]

hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

input = wlrec[0]
true = [hurt[0], win[0], sad[0]]
alpha = 0.1


# results_my = nn(input=input, weight=weight, goal_prediction=true, alpha=alpha)

for i in range(5):
    print(f'\n\nITER = {i}')
    # prediction, delta, error, derivative, weights
    results_my = nn(input=input, weight=weight, goal_prediction=true, alpha=alpha)
    weight = results_my[4]

