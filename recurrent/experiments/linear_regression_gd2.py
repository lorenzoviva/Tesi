#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *
import visdom as vis
import time
v = vis.Visdom()
# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    plot_all(points, b_gradient, m_gradient, b_current, m_current, new_b, new_m)
    print("mean error:" + str(b_gradient/2))
    return [new_b, new_m]

def plot_all(points, b_gradient, m_gradient, b_current, m_current, new_b, new_m):
    lines_x_interval = [-100,100]
    lines_y_boundaries = [-100,100]
    gradient_line_x = current_line_x = new_line_x = lines_x_interval
    gradient_line_y = [x*m_gradient + b_gradient for x in gradient_line_x]


    current_line_y = [x*m_current + b_current for x in current_line_x]
    new_line_y = [x*new_m + new_b for x in new_line_x]
    data = [{
        'x': gradient_line_x,
        'y': gradient_line_y,
        'type':'line',
        'mode':"lines",
        'color':"blue",
        'label':["grad"],

    },{
        'x': current_line_x,
        'y': current_line_y,
        'type':'line',
        'mode':"lines",
        'color':"green",
        'text':["curr"],
    },{
        'x': new_line_x,
        'y': new_line_y,
        'type':'line',
        'mode':"lines",
        'color':"red",
        'text':["new"],
    },{
        'x': list(points[:,0]),
        'y': list(points[:,1]),
        'type': 'scatter',
        'mode': 'markers',
        'width':800,
        'height':800,
    }]


    win = 'mytestwin'
    env = 'main'

    layout= {
        'title':"Test Plot",
        'xaxis':{'title':'x1'},
        'yaxis':{'title':'x2'}
    }
    opts = {}

    fig = v._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})
    v.update_yaxes(range=[-100, 100])
    time.sleep(4)

def clip_y(y_array,x_array, lines_y_boundaries, b, m):
    if max(y_array) > max(lines_y_boundaries):
        max_y = max(y_array)
        new_y = max(lines_y_boundaries)
        new_x = (new_y - b) / float(m)
        for i,y in enumerate(y_array):
            if max_y == y:
                y_array[i] = new_y
                x_array[i] = new_x

    if min(y_array) < min(lines_y_boundaries):
        min_y = min(y_array)
        new_y = min(lines_y_boundaries)
        new_x = (new_y - b) / float(m)
        for i,y in enumerate(y_array):
            if min_y == y:
                y_array[i] = new_y
                x_array[i] = new_x

    return y_array,x_array


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()