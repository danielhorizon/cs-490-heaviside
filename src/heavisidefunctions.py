import math
import tensorflow as tf


def sigmoid_heaviside(k=1):
    '''
        Simple sigmoid
            - limits do not necessarily converge to the heaviside function
            - derivative can be 0
        = a/(1+e^(-k*(x-threshold)))
        for simplicity: a == 1
    '''
    def f(xs_thresholds):
        x, ti = xs_thresholds
        # shift to threshold
        x = x - ti
        return 1/(1+tf.exp(-k*x))
    return f

# fit sigmoid helpers


def threshold_input_to_threshold_param(x, w=math.pi, o=math.pi/2):
    return -1 * tf.math.tan(x*w-o) + 0.5


def lin(x, m, b):
    return m*x+b


def threshold_param_to_weight(x, neg_w_b=[-20.00448894, 23.74890989], pos_w_b=[20.00475821, 3.7403742]):
    ''' There is a linear mapping between these params'''
    return tf.where(x < 0.5, lin(x, *neg_w_b), lin(x, *pos_w_b))


def fit_sigmoid_heaviside(xs_thresholds):
    '''
        fit to our linear approximation using least squares
        http://jpeg:9998/notebooks/notebooks/Activation%20Function%20Plots.ipynb#Continuous-Functions
    '''
    x, ti = xs_thresholds
    t = threshold_input_to_threshold_param(ti)
    weight = threshold_param_to_weight(t)
    multiplier = 10
    s = 1/(1+tf.math.exp(-x*weight+weight/2+(multiplier * (0.5-t))))
    return s


def linear_heaviside(xs_thresholds, delta=0.1):
    ''' piecewise linear approximation of the heaviside function
        x_and_t: x values for a given threshold
    '''
    x, t = xs_thresholds
    d = delta
    tt = tf.minimum(t, 1-t)
    m1 = d/(t-tt/2)
    m2 = (1-2*d)/tt
    m3 = d/(1-t-tt/2)
    return tf.where(x < t - tt/2, m1*x,
                    tf.where(x > t + tt/2, m3*x + (1-d-m3*(t+tt/2)),
                             m2*(x-t)+0.5))
