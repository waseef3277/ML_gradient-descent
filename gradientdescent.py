import matplotlib.pyplot as plt
import numpy as np

# y = mx + b
# m is slope, b is y-intercept
"""
np.random.seed(50)
num = 25
ranges = np.array([[10, 55], [25, 60]])
x = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(num, ranges.shape[0]))
print(x)
np.savetxt('housing.csv', x, fmt='%.2f', delimiter=',', header="area,price")
"""

def get_error(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def gradient_step(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    n = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/n) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/n) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = gradient_step(b, m, np.array(points), learning_rate)
    return [b, m]


def run():
    points = np.genfromtxt("housing.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 100000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m,
                                                                              get_error(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m,
                                                                      get_error(b, m, points)))


if __name__ == '__main__':
    run()

