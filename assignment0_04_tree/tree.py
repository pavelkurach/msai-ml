import numpy as np
from sklearn.base import BaseEstimator

EPS = 0.000000005

def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    if len(y) == 0: return 0
    probs = np.mean(y, axis=0)
    return -np.sum(probs * np.log(probs + EPS))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    if len(y) == 0: return 0
    probs = np.mean(y, axis=0)
    return 1 - np.sum(probs * probs)


def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    if len(y) == 0: return 0

    return np.std(y)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    if len(y) == 1: return 0
    if len(y) == 2: return np.abs(y[0] - y[1]) / 2

    return np.mean(np.absolute(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self,
                 feature_index,
                 threshold,
                 proba=0,
                 left_child=None,
                 right_child=None):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = left_child
        self.right_child = right_child


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the provided subset where selected feature x^j >= threshold
        """

        X_left = X_subset[X_subset[:, feature_index] < threshold]
        y_left = y_subset[X_subset[:, feature_index] < threshold]

        X_right = X_subset[X_subset[:, feature_index] >= threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        y_left = y_subset[X_subset[:, feature_index] < threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        if len(np.unique(y_subset, axis=0)) <= 1:
            return None, None
        Q, n_features = X_subset.shape
        G_min = 1 / EPS
        feature_index, threshold = 0, X_subset[0, 0]

        for feature in range(n_features):
            for value in np.unique(X_subset[:, feature]):
                y_left, y_right = self.make_split_only_y(
                    feature_index=feature,
                    threshold=value,
                    X_subset=X_subset,
                    y_subset=y_subset,
                )
                G = self.criterion(y_left) * len(y_left) / Q + \
                    self.criterion(y_right) * len(y_right) / Q
                if G < G_min:
                    G_min = G
                    feature_index, threshold = feature, value

        return feature_index, threshold

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        feature_index, threshold = self.choose_best_split(X_subset=X_subset,
                                                          y_subset=y_subset)

        if (feature_index is None or threshold is None) or \
                self.max_depth is not None and self.depth == self.max_depth:
            self.depth -= 1
            return Node(None, None, np.mean(y_subset, axis=0))

        self.depth += 1

        (X_left, y_left), (X_right, y_right) = self.make_split(
            feature_index=feature_index,
            threshold=threshold,
            X_subset=X_subset,
            y_subset=y_subset,
        )

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left_child=self.make_tree(X_left, y_left),
            right_child=self.make_tree(X_right, y_right)
        )

    def fit(self, X, y):

        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[
            self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)


    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        n_objects = len(X)
        y_predicted = np.empty((n_objects, 1))

        def predict_ith(X_i):
            node: Node = self.root
            while True:
                feature_index = node.feature_index
                threshold = node.value
                if not threshold is None and X_i[feature_index] < threshold:
                    node = node.left_child
                elif not threshold is None and X_i[feature_index] >= threshold:
                    node = node.right_child
                else:
                    if not self.classification:
                        return node.proba
                    return np.argmax(node.proba)

        for i in range(n_objects):
            y_predicted[i] = predict_ith(X[i, :])

        return y_predicted


    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        n_objects = len(X)
        y_predicted_probs = np.empty((n_objects, self.n_classes))

        def predict_ith(X_i):
            node: Node = self.root
            while True:
                feature_index = node.feature_index
                threshold = node.value
                if not threshold is None and X_i[feature_index] < threshold:
                    node = node.left_child
                elif not threshold is None and X_i[feature_index] >= threshold:
                    node = node.right_child
                else:
                    return node.proba

        for i in range(n_objects):
            y_predicted_probs[i] = predict_ith(X[i, :])

        return y_predicted_probs

