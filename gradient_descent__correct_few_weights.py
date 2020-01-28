import numpy as np

input_f = input


# Градиентный спуск
def gradient_descent(inputs, weights, goal_pred, alpha):
    assert(len(inputs) == len(weights))
    prediction, delta, error, derivative = 0, 0, 0, 0
    weights = weights
    print(f'inputs == {inputs}')
    # print(f'weights == {weights}')
    for i in range(len(inputs)):
        prediction = inputs[i] * weights[i]   # Предсказанная величина
        print(f'prediction = {prediction}')
        delta = prediction - goal_pred  # Чистая ошибка
        print(f'delta = {delta}')
        error = delta ** 2              # Среднеквадратичная ошибка
        print(f'error = {error}')
        derivative = delta * inputs[i]     # Производная
        print(f'derivative = {derivative}')
        weights[i] -= derivative * alpha   # Изменение весовых коэффициенитов
        input_f()
    return error, weights, derivative, delta, prediction


def neural_network(input, weights, learning_rate, goal_predictions):
    error, weights, derivative, delta, prediction = gradient_descent(inputs=input,
                                                                     weights=weights,
                                                                     alpha=learning_rate,
                                                                     goal_pred=goal_predictions)
    print(f'Error = {error}, prediction = {prediction}, goal prediction = {goal_predictions},'
          f'\nweights = {weights} \n'
          f'derivative = {derivative}, delta = {delta}\n')
    return weights

# def ele_mul(number, vector):
#     output = [0, 0, 0]
#     assert(len(output) == len(vector))
#     for i in range(len(vector)):
#         output[i] = number * vector[i]
#     return output

toes  = [8.5,  9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2,  1.3, 0.5, 1.0]

win_or_lose_binary = np.array([1, 1, 0, 1])  # win, win, lose, win

true = np.array(win_or_lose_binary[0])
input = np.array([toes[0], wlrec[0], nfans[0]])

weights = np.array([0.1, 0.2, -0.1])
alpha = 1
# inputs = np.array([2])
# weight = np.array([0.5])
# goal_pred = np.array([0.8])
# alpha = np.array([0.1])

inputs = input
weight = weights
goal_pred = true
alpha = alpha

for i in range(20):
    weight = weight
    print(f'Iter = {i}')
    weight = neural_network(input=inputs, weights=weight, learning_rate=alpha, goal_predictions=goal_pred)

