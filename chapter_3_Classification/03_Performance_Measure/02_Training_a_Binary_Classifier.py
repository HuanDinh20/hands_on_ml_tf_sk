from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np

digits = load_digits()
X, Y = digits['data'], digits['target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

"""
Let’s also shuffle the training set; this will guarantee that all cross-validation folds will
be similar (you don’t want one fold to be missing some digits). Moreover, some learn‐
ing algorithms are sensitive to the order of the training instances, and they perform
poorly if they get many similar instances in a row. Shuffling the dataset ensures that
this won’t happen
note: Shuffling may be a bad idea in some contexts—for example, if you are working on time series data (such as
stock market prices or weather conditions). We will explore this in the next chapters.

"""

shuffle_index = np.random.permutation(len(y_train))

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

"""
Let’s simplify the problem for now and only try to identify one digit—for example,
the number 5. This “5-detector” will be an example of a binary classifier, capable of
distinguishing between just two classes, 5 and not-5.

"""
y_train_5 = (y_train == 5)  # True if 5, False if other
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


sgd_clf.predict(X[5], Y[5])