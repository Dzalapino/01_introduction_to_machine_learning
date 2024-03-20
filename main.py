from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

feature_dict = {
    0: 'sepal length',
    1: 'sepal width',
    2: 'petal length',
    3: 'petal width'
}

iris_dataset = datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target


def visualise_data(feature_x: int, feature_y: int) -> None:
    print(f'\nPlotting the {feature_dict[feature_x]} and {feature_dict[feature_y]} of the iris dataset')
    x_min, x_max = X[:, feature_x].min() - .5, X[:, feature_x].max() + .5
    y_min, y_max = X[:, feature_y].min() - .5, X[:, feature_y].max() + .5

    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, feature_x], X[:, feature_y], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(feature_dict[feature_x])
    plt.ylabel(feature_dict[feature_y])

    legend1 = ax.legend(*scatter.legend_elements(), loc='lower right', title='Class label')
    ax.add_artist(legend1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


def train_and_visualize_classifier(X_train, y_train, filename=None) -> LogisticRegression:
    # linear binary classification with a logistic regression model
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    # read the logistic regression model parameters
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T

    # calculate the intercept and gradient of the decision boundary
    c = -b / w2
    m = -w1 / w2

    # plot the data and the classification boundary
    x_min, x_max = np.min(X_train, 0)[0] - 1, np.max(X_train, 0)[0] + 1
    y_min, y_max = np.min(X_train, 0)[1] - 1, np.max(X_train, 0)[1] + 1

    xd = np.array([x_min, x_max])
    yd = m * xd + c

    plt.figure()
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, y_min, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, y_max, color='tab:orange', alpha=0.2)

    # plot the training and test data
    plt.scatter(*X_train.T, c=y_train, cmap=plt.cm.Set1, edgecolor='k')
    plt.scatter(*X_test.T, c=y_test, cmap=plt.cm.Set1, edgecolor='b')

    # plot the decision boundary
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(r'sepal length')
    plt.ylabel(r'sepal width')
    plt.title('Logistic Regression Decision Boundary')
    if filename is not None:
        plt.savefig(filename + '.png')
    plt.show()

    return clf


print('Checking dimensions of X and y arrays')
print(X.shape)
print(y.shape)

visualise_data(0, 1)
print('From the given plot we can easily see that only the iris-setosa is easily separable from the other two classes')

visualise_data(2, 3)
print('From the given plot we can easily see that all the classes are separable from each other\n'
      '(with some small overlap between the iris-versicolor and iris-virginica)')

visualise_data(0, 3)
print('As in the previous example')

visualise_data(2, 1)
print('As in the previous example')

visualise_data(0, 2)
print('As in the previous example')

visualise_data(1, 3)
print('As in the previous example')

# indices 0-49, 50-99, 100-149 in random order
random0 = np.random.choice(np.arange(0, 50), 50, replace=False)
random1 = np.random.choice(np.arange(50, 100), 50, replace=False)
random2 = np.random.choice(np.arange(100, 150), 50, replace=False)

# take 80% of each class for training
X0 = X[random0[:40], :]
X1 = X[random1[:40], :]
X2 = X[random2[:40], :]

# take the coresponding labels
y0 = y[random0[:40]]
y1 = y[random1[:40]]
y2 = y[random2[:40]]

# take the remaining 20% of each class for testing
X0_test = X[random0[40:], :]
X1_test = X[random1[40:], :]
X2_test = X[random2[40:], :]

# take the coresponding labels
y0_test = y[random0[40:]]
y1_test = y[random1[40:]]
y2_test = y[random2[40:]]

# compose the training set (just features 0 and 1 and classes 0 and 1)
X_train = np.concatenate((X0[:, 0:2], X1[:, 0:2]))
y_train = np.concatenate((y0, y1))
X_test = np.concatenate((X0_test[:, 0:2], X1_test[:, 0:2]))
y_test = np.concatenate((y0_test, y1_test))

train_and_visualize_classifier(X_train, y_train)  # train the classifier and visualize the decision boundary

# compose the training set (just features 0 and 1 and classes 1 and 2)
X_train = np.concatenate((X1[:, 0:2], X2[:, 0:2]))
y_train = np.concatenate((y1, y2))
X_test = np.concatenate((X1_test[:, 0:2], X2_test[:, 0:2]))
y_test = np.concatenate((y1_test, y2_test))

clf = train_and_visualize_classifier(X_train, y_train, 'virginica_versicolor3')  # Again for the other two classes

# predict the classes of the training and test samples
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# count the number of misclassified samples in both the training and test sets
misclassified_train = (y_train != y_train_pred).sum()
misclassified_test = (y_test != y_test_pred).sum()

print(f'Number of misclassified training samples: {misclassified_train}')
print(f'Number of misclassified test samples: {misclassified_test}')

"""
In machine learning, the decision boundary is determined based on the training data. 
If the training data changes, the decision boundary may also change. 
This is because the classifier is trying to find the best boundary that separates the classes in the training data. 
If the training data is different, the "best" boundary may also be different.
Similarly, the classification results can depend on the training set.
If the classifier is trained on different data, it may make different predictions for the same test data.
This is because the classifier has learned different patterns from the different training data.
"""
