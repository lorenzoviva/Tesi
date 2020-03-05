#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *
import visdom as vis
import time
v = vis.Visdom()
# y = mx + b
# m is slope, b is y-intercept
grd_m = []
grd_b = []
grd_space_x = []
grd_space_y = []
grd_space_z = []
grd_space_lines = []
max_points=10


def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate, iteration, superstep=True ):
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
    if superstep:
        grd_m.append([learningRate*m_gradient,m_current])
        grd_b.append([learningRate*b_gradient,b_current])
        grd_space_x.append(m_current)
        grd_space_y.append(b_current)
        grd_space_lines.append([m_current,new_m,b_current,new_b])
        grd_space_z.append(compute_error_for_line_given_points(b_current, m_current, points))
        b = 0
        m = 0
        for i in range(10000):
            b, m = step_gradient(b, m, array(grd_space_y + grd_space_x).reshape((-1,2))[-max_points:,:], 0.001, iteration, False)
        plot_all(points, learningRate*b_gradient, learningRate*m_gradient, b_current, m_current, new_b, new_m, b ,m, iteration)
    print(("" if superstep else "--------") + "mean error:" + str(b_gradient) + " m gradient: " + str(m_gradient))
    return [new_b, new_m]


def plot_all(points, b_gradient, m_gradient, b_current, m_current, new_b, new_m, b, m, iteration):
    lines_x_interval = [-100,100]
    gradient_line_x = current_line_x = new_line_x = lines_x_interval
    gradient_line_y = [x*m_gradient + b_gradient for x in gradient_line_x]
    current_line_y = [x*m_current + b_current for x in current_line_x]
    new_line_y = [x*new_m + new_b for x in new_line_x]
    data = [{
        'x': gradient_line_x,
        'y': gradient_line_y,
        'type': 'line',
        'mode': "lines",
        'color': "blue",
        'label': "grad",
    },{
        'x': current_line_x,
        'y': current_line_y,
        'type': 'line',
        'mode': "lines",
        'color': "green",
        'text': "curr",
    },{
        'x': new_line_x,
        'y': new_line_y,
        'type': 'line',
        'mode': "lines",
        'color': "red",
        'text': ["new"],
    },{
        'x': list(points[:, 0]),
        'y': list(points[:, 1]),
        'type': 'scatter',
        'mode': 'markers',
    }]

    win = 'mainplotwin'
    env = 'main'

    layout = {
        'title':"Main Plot " + str(iteration),
        'xaxis':{'title':'x1'},
        'yaxis':{'title':'x2','range':[-100, 100]}
    }
    opts = {}

    v._send({'data': data, 'win': win, 'eid': env, 'layout': layout, 'opts': opts})


    b_lines_x = lines_x_interval
    b_lines_y = [x*m + b for x in b_lines_x]

    grad_space_win = 'grad_space_win'
    grad_space_data = []
    # for [X1,X2,Y1,Y2] in grd_space_lines:
    #     grad_space_data.append({'x': [X1,X2],
    #                             'y': [Y1,Y2],
    #                             'type': 'line',
    #                             'mode': "lines",
    #                             'color': "blue",
    #                             'label': "grad",})
    grad_space_data.append({
        'x': list(grd_space_y),
        'y': list(grd_space_x),
        'z': list(grd_space_z),
        'type': 'scatter3d',
        'mode': 'markers',
    })
    grad_space_data.append({
        'x': b_lines_x,
        'y': b_lines_y,
        'z': [0 for _ in b_lines_x],
        'type': 'scatter3d',
        'mode': "lines",
        'color': "blue",
        'label': "grad",
    })
    grad_space_layout = {
        'title': "Gradients Plot",
        'scene': {
            'xaxis': {'title':'m','range':[-1, 2]},
            'yaxis': {'title':'b','range':[-1, 1]},
            'zaxis': {'title':'e','range':[0, 50]}
        }
    }
    # grad_space_layout = {
    #     'title':"Main Plot",
    #     'xaxis':{'title':'m'},#,'range':[-1, 2]},
    #     'yaxis':{'title':'b'}#'range':[-1, 1]}
    # }
    v._send({'data': grad_space_data, 'win': grad_space_win, 'eid': env, 'layout': grad_space_layout, 'opts': opts})

    #
    # grad_m_win = 'gradmwin'
    # v.scatter(grd_m,win=grad_m_win,opts=dict({"title":"m gradient vs m"}))
    #
    # grad_b_win = 'gradbwin'
    # v.scatter(grd_b,win=grad_b_win,opts=dict({"title":"b gradient vs b"}))
    # time.sleep(1)


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
        b, m = step_gradient(b, m, array(points), learning_rate, i)

    return [b, m]

def run():
    points = genfromtxt("data3.csv", delimiter=",")
    learning_rate = 0.0002
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 10000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()