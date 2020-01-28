import numpy as np


# Градиентный спуск
def gradient_descent(inputs, weights, goal_pred, alpha):
    prediction = inputs * weights   # Предсказанная величина
    delta = prediction - goal_pred  # Чистая ошибка
    error = delta ** 2              # Среднеквадратичная ошибка
    derivative = delta * inputs     # Производная
    weights -= derivative * alpha   # Изменение весовых коэффициенитов
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


inputs = np.array([2])
weight = np.array([0.5])
goal_pred = np.array([0.8])
alpha = np.array([0.1])

inputs = inputs[0]
weight = weight[0]
goal_pred = goal_pred[0]
alpha = alpha[0]

for i in range(20):
    weight = weight
    print(f'Iter = {i}')
    weight = neural_network(input=inputs, weights=weight, learning_rate=alpha, goal_predictions=goal_pred)
