import numpy as np

class Log_Reg():
    def __init__(self, max_iter = 1000, learning_rate = 0.01, method = "grad_dec"):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.method = method
        self.weights = None

    def sigmoid_func(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = self.addition_term(X)
        self.weights = np.zeros(X.shape[1])
        if self.method == 'grad_dec':
            self.grad_dec(X, y)
        else:
            self.newton_opt(X, y)

    def addition_term(self, X):
        term = np.ones((X.shape[0], 1))
        return np.concatenate((term, X), axis=1)

    def get_probs(self, X):
        X = self.addition_term(X)
        z = np.matmul(X, self.weights)
        return self.sigmoid_func(z)

    def predict(self, X):
        X = self.addition_term(X)
        z = np.matmul(X, self.weights)

        return (self.sigmoid_func(z) > 0.5).astype(int)

    def grad_dec(self, X, y):
        for i in range(self.max_iter):
            z = np.matmul(X, self.weights)
            preds = self.sigmoid_func(z)
            #транспонируем матрицу хар-тик, каждое значение икса в строке умножаем на разность прогноза и факта
            gradient = np.matmul(X.T, (preds - y)) / X.shape[1]
            prev_weights = self.weights
            self.weights = self.weights - self.learning_rate * gradient
            if np.sum(abs(self.weights - prev_weights)) < 1e-6:
                break

    def hessian_cnt(self, X):
        z = np.matmul(X, self.weights)
        h = self.sigmoid_func(z)
        R = np.diag(h * (1 - h))
        return np.matmul(X.T, np.matmul(R, X))
    def newton_opt(self, X, y):
        for i in range(self.max_iter):
            z = np.matmul(X, self.weights)
            preds = self.sigmoid_func(z)
            gradient = np.matmul(X.T, (preds - y)) / X.shape[1]
            hessian = self.hessian_cnt(X)
            self.weights = self.weights - self.learning_rate * np.linalg.inv(hessian) @ gradient
            prev_weights = self.weights
            self.weights -= np.linalg.inv(hessian) @ gradient

            if np.sum(abs(self.weights - prev_weights)) < 1e-6:
                break



def loss_func(probabilities, Y):
    tmp = 0
    epsilon = 1e-15  #небольшой эпсилон для логарифма
    for p, y in zip(probabilities, Y):
        p = np.maximum(epsilon, np.minimum(1 - epsilon, p))  #применяем ограничение к p
        tmp += (y * np.log(p) + (1 - y) * np.log(1 - p))
    log_loss = - (1 / len(Y)) * tmp
    return log_loss
