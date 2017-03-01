#MEHMET TUGRUL SAVRAN
#CREATED FEBRUARY, 2017 

import numpy as np 


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    if label*(np.dot(current_theta,feature_vector)+current_theta_0) <= 0:
        return (current_theta + label*feature_vector, current_theta_0 + label)
    return (current_theta,current_theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    k = len(feature_matrix)
    num_cols = len(feature_matrix[0])
    theta = np.zeros(num_cols)
    theta_0 = 0.0 
    for t in range(T):
        for i in np.random.permutation(k):
            if t == 0 and i == 0:
                theta = labels[i]*feature_matrix[i]
                theta_0 += labels[i]
            else:
                (theta,theta_0) = perceptron_single_step_update(feature_matrix[i],labels[i],theta,theta_0)
    return (theta, theta_0) 















