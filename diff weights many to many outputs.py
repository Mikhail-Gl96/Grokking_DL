import numpy as np


def get_prediction(input, weight):
    assert (len(input) == len(weight))
    prediction_value = [0 for i in range(len(input))]
    for i in range(len(weight)):
        prediction_value[i] += sum(input * weight[i])
    return prediction_value


def get_delta(prediction, goal_prediction):
    delta_value = np.array([prediction[i] - goal_prediction[i] for i in range(len(goal_prediction))])
    print('')
    for i in range(len(goal_prediction)):
        print(f'*--> [prediction[{i}] <{prediction[i]}> - goal_prediction[{i}] <{goal_prediction[i]}> = {prediction[i] - goal_prediction[i]} delta_value ')
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
    print('\n\n')
    print(f'input = {input}\n'
          f'prediction = {prediction}\n'
          f'error = {error}\n'
          f'delta = {delta}\n'
          f'derivative(weight_deltas) = {derivative}\n'
          f'weights = {weights}')
    print('- '*50)
    print(f'input = {input}\n'
          f'prediction = {prediction}\n'
          f'error = {error}\n'
          f'delta = {delta}\n'
          f'derivative(weight_deltas) = {[i[1] for i in derivative]}\n'  # производная для случая индекс 1
          f'weights = {weights}')
    return prediction, delta, error, derivative, weights


def test_nn(input, weight, goal_prediction):
    prediction = get_prediction(input=input, weight=weight)
    delta = get_delta(prediction=prediction, goal_prediction=goal_prediction)
    error = get_error(delta=delta)
    print(f'input = {input}\n'
          f'prediction = {prediction}  goal prediction = {goal_prediction}\n'
          f'error = {error}\n'
          f'delta = {delta}\n')


# игр %победа # болельщики
weight_hurt = np.array([0.1, 0.1, -0.3])
weight_win = np.array([0.1, 0.2, 0.0])
weight_sad = np.array([0.0, 1.3, 0.1])

weight = [weight_hurt.copy(),  # травмы?
          weight_win.copy(),   # победа?
          weight_sad.copy()]   # печаль?


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])


hurt = np.array([0.1, 0.0, 0.0, 0.1])
win = np.array([1, 1, 0, 1])
sad = np.array([0.1, 0.0, 0.1, 0.2])

alpha = 0.01

input = np.array([toes[0], wlrec[0], nfans[0]])
true = np.array([hurt[0], win[0], sad[0]])


# results_my = nn(input=input, weight=weight, goal_prediction=true, alpha=alpha)

for i in range(1):
    print(f'\n\nITER = {i}')
    # prediction, delta, error, derivative, weights
    results_my = nn(input=input, weight=weight, goal_prediction=true, alpha=alpha)
    weight = results_my[4]

# print("\n\n\n  TEST")
# index = 0
# input = [toes[index], wlrec[index], nfans[index]]
# true = [hurt[index], win[index], sad[index]]
# test_nn(input=input, weight=weight, goal_prediction=true)
