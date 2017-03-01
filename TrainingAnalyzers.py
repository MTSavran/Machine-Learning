#MEHMET TUGRUL SAVRAN
#CREATED FEBRUARY, 2017 

"""This file includes fundamental 
analysis tools for the accuracy of ML implementations. It collects the 
analyzers of each classifier in the Machine Learning repository. """

import numpy as np 
import perceptron as p1 
import average_perceptron as p2 
import pegasos as p3 

def accuracy(classifications, targets):
    """
    Given equal length vectors containing predicted and actual labels,
    returns the ratio of correct predictions.
    """
    return (classifications == targets).mean()


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0.
    """
    (samples_number, features_number) = feature_matrix.shape
    classifications = np.zeros(samples_number)
    for i in range(samples_number):
        feature_vector = feature_matrix[i]
        classification = np.dot(theta, feature_vector) + theta_0
        if (classification > 0):
            classifications[i] = 1
        else:
            classifications[i] = -1
    return classifications

def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    
    (trained_theta, trained_theta_0) = perceptron(train_feature_matrix, train_labels, T)
    train_classifications  = classify(train_feature_matrix, trained_theta, trained_theta_0)
    data_classifications = classify(val_feature_matrix,trained_theta,trained_theta_0)
    train_correctness = accuracy(train_classifications, train_labels)
    data_correctness = accuracy(data_classifications, val_labels)
    return (train_correctness, data_correctness)



def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    theta, theta_0 = average_perceptron(train_feature_matrix, train_labels, T)
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_predictions, train_labels)
    val_accuracy = accuracy(val_predictions, val_labels)
    return (train_accuracy,val_accuracy)


def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Trains a linear classifier using the pegasos algorithm
    with given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    (theta, theta_0) = pegasos(train_feature_matrix, train_labels, T, L)
    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_predictions, train_labels)
    val_accuracy = accuracy(val_predictions, val_labels)
    return (train_accuracy,val_accuracy)





