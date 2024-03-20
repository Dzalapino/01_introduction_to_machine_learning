import time

from sklearn import datasets
import matplotlib.pyplot as plt


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


print('Checking dimensions of X and y arrays')
print(X.shape)
print(y.shape)

visualise_data(0, 1)
print('From the given plot we can easily see that only the iris-setosa is easily separable from the other two classes')
time.sleep(3)

visualise_data(2, 3)
print('From the given plot we can easily see that all the classes are separable from each other\n'
      '(with some small overlap between the iris-versicolor and iris-virginica)')
time.sleep(3)

visualise_data(0, 3)
print('As in the previous example')
time.sleep(3)

visualise_data(2, 1)
print('As in the previous example')
time.sleep(3)

visualise_data(0, 2)
print('As in the previous example')

visualise_data(1, 3)
print('As in the previous example')
time.sleep(3)
