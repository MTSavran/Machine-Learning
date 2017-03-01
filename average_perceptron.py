#MEHMET TUGRUL SAVRAN
#CREATED FEBRUARY, 2017

import numpy as np 
import perceptron as p1

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    k = len(feature_matrix)
    num_cols = len(feature_matrix[0])
    theta = np.zeros(num_cols)
    theta_0 = 0.0 
    sum_theta = theta 
    sum_theta_0 = theta_0 
    for t in range(T):
        for i in np.random.permutation(k):
            if t == 0 and i == 0:
                theta = labels[i]*feature_matrix[i]
                theta_0 += labels[i]
            else:
                (theta,theta_0) = p1.perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
            sum_theta += theta 
            sum_theta_0 += theta_0
    size = float(k*T)
    theta = sum_theta * 1/size 
    theta_0 = sum_theta_0 * 1/size
    return (theta, theta_0)