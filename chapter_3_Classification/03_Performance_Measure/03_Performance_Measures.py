"""
Evaluating a classifier is often significantly trickier than evaluating a regressor, so we
will spend a large part of this chapter on this topic. There are many performance
measures available, so grab another coffee and get ready to learn many new concepts
and acronyms!

************* Measuring Accuracy Using Cross-Validation *******************

Let’s use the cross_val_score() function to evaluate your SGDClassifier model
using K-fold cross-validation, with three folds. Remember that K-fold crossvalidation means splitting the training set into K-folds (in this case, three), then mak‐
ing predictions and evaluating them on each fold using a model trained on the
remaining folds
"""
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

digits = load_digits()
X, Y = digits['data'], digits['target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
shuffle_index = np.random.permutation(len(y_train))

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)  # True if 5, False if other
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print('Predict 5 accuracy: ')
print(cross_val_score(sgd_clf, X_train, y_train_5, ))
"""
Wow! Above 98% accuracy (ratio of correct predictions) on all cross-validation folds? 
This looks amazing, doesn’t it? Well, before you get too excited, let’s look at a very
dumb classifier that just classifies every single image in the “not-5” class:
"""
from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X_t, y=None):
        pass

    def predict(self, X_t):
        return np.zeros((len(X_t), 1), dtype=bool)


never_5_clf = Never5Classifier()
never_5_score = cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print('*' * 20)
print('never predict 5 accuracy: ')
print(never_5_score)
"""
That's right: it has over 88 accuracy, 
This is simply because only about 10% of the
images are 5s, so if you always guess that an image is not a 5, you will be right about
90% of the time. Beats Nostradamus.
This demonstrates why accuracy is generally not the preferred performance measure
for classifiers, especially when you are dealing with skewed datasets (i.e., when some
classes are much more frequent than others).
"""

"""
***************************** Confusion Matrix  *********************************
A much better way to evaluate the performance of a classifier is to look at the confu‐
sion matrix. EachTo compute the confusion matrix, you first need to have a set of predictions, so they
can be compared to the actual targets. You could make predictions on the test set, but
let’s keep it untouched for now (remember that you want to use the test set only at the
very end of your project, once you have a classifier that you are ready to launch).

To compute the confusion matrix, you first need to have a set of predictions, so they
can be compared to the actual targets. 
s. You could make predictions on the test set, but
let’s keep it untouched for now (remember that you want to use the test set only at the
very end of your project, once you have a classifier that you are ready to launch).
"""

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

"""
Just like the cross_val_score() function, cross_val_predict() performs K-fold
cross-validation, but instead of returning the evaluation scores, it returns the predic‐
tions made on each test fold. This means that you get a clean prediction for each
instance in the training set (“clean” meaning that the prediction is made by a model
that never saw the data during training).
Now you are ready to get the confusion matrix using the confusion_matrix() func‐
tion.
"""

confusionMatrix = confusion_matrix(y_train_5, y_train_pred)

print('>>' * 10)
print('Confusion Matrix for SGD binary classifier: ')
print(confusionMatrix)

"""
the result is :
[[1134    7]
 [   9  107]]
Each row in a confusion matrix represents an actual class, while each column repre‐
sents a predicted class.
The first row of this matrix considers non-5 images (the nega‐
tive class): 1134 of them were correctly classified as non-5s (they are called true
negatives), while the remaining 7 were wrongly classified as 5s (false positives)
The second row considers the images of 5s (the positive class): 9 were wrongly
classified as non-5s (false negatives), while the remaining 107 were correctly classi‐
fied as 5s (true positives).
        Predict
Actual  TN       FP
        FN       TP

a perfect confusion matrix:
        100      0
        0        40
        
The confusion matrix gives you a lot of information, but sometimes you may prefer a
more concise metric. An interesting one to look at is the accuracy of the positive pre‐
dictions; this is called the precision of the classifier:

precision = TP / ( TP + FP)

TP is the number of true positives, and FP is the number of false positives.

A trivial way to have perfect precision is to make one single positive prediction and
ensure it is correct (precision = 1/1 = 100%).
This would not be very useful since the
classifier would ignore all but one positive instance.

So precision is typically used along with another metric named 
recall, also called sensitivity or true positive rate 
(TPR): this is the ratio of positive instances that are correctly detected by the classifier

recall = TP / ( TP + FN)

FN is of course the number of false negatives.

********************************* Precision and Recall ********************************

"""
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)

print('<>' * 10)
print('Precision: ')
print(precision)
print('Recall: ')
print(recall)

"""
F1 Score:
 - It is often convenient to combine precision and recall into a single metric called the F1
score, in particular if you need a simple way to compare two classifiers.
 - The F1 score is the harmonic mean of precision and recall 
 Whereas the regular mean treats all values equally, the harmonic mean gives much more weight to low values.
 As a result, the classifier will only get a high F1 score if both recall and precision are high.
 
 
            F1 = 2 /( ( 1 / precision ) + ( 1 / recall ) )
            
"""
from sklearn.metrics import f1_score

f1 = f1_score(y_train_5, y_train_pred)
print('F1 Score: ')
print(f1)

"""
The F1
score favors classifiers that have similar precision and recall. This is not always
what you want: in some contexts you mostly care about precision, and in other con‐
texts you really care about recall.
For example, if you trained a classifier to detect videos that are safe for kids, 
you would probably prefer a classifier that rejects many good videos (low recall) 
but keeps only safe ones (high precision), 
On the other hand, suppose you train a classifier to detect
shoplifters on surveillance images: it is probably fine if your classifier has only 30%
precision as long as it has 99% recall (sure, the security guards will get a few false
alerts, but almost all shoplifters will get caught).
Unfortunately, you can’t have it both ways: increasing precision reduces recall, and
vice versa. This is called the precision/recall tradeoff.


*********************************** Precision/Recall Tradeoff ***********************

To understand this tradeoff, let’s look at how the SGDClassifier makes its classifica‐
tion decisions. For each instance, it computes a score based on a decision function, 
and if that score is greater than a threshold, it assigns the instance to the positive
class, or else it assigns it to the negative class.

Scikit-Learn does not let you set the threshold directly, but it does give you access to
the decision scores that it uses to make predictions. Instead of calling the classifier’s
predict() method, you can call its decision_function() method, which returns a
score for each instance, and then make predictions based on those scores using any
threshold you want:

"""
y_scores = sgd_clf.decision_function(X_train[:3])
print('*' * 10)
print('y test', y_train_5[:3])
print('Tradeoff :')
print('y_scores: ')
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print('After compare with threshold == 0')
print('y pred: ', y_some_digit_pred)

"""
The SGDClassifier uses a threshold equal to 0, so the previous code returns the same
result as the predict() method (i.e., True).
"""
threshold = -2000
y_some_digit_pred = (y_scores > threshold)
print('After compare with threshold == -2000')
print('y pred: ', y_some_digit_pred)

"""
Decrease threshold increase recall, and vice versa

So how can you decide which threshold to use? For this you will first need to get the
scores of all instances in the training set using the cross_val_predict() function
again, but this time specifying that you want it to return decision scores instead of
predictions:

"""

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

"""
If someone says “let’s reach 99% precision,” you should ask, “at
what recall?”

"""

"""
************************** The ROC Curve *****************************

The receiver operating characteristic (ROC) curve is another common tool used with
binary classifiers. It is very similar to the precision/recall curve, but instead of plot‐ting precision versus recall, 
the ROC curve plots the true positive rate (another name for recall) against the false positive rate. 
 The FPR is the ratio of negative instances that are incorrectly classified as positive. 
 It is equal to one minus the true negative rate which is the ratio of negative instances that are correctly classified
  as negative
  The
TNR is also called specificity. Hence the ROC curve plots sensitivity (recall) versus
1 – specificity.
"""

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


plot_roc_curve(fpr, tpr)
plt.show()

"""
Once again there is a tradeoff: the higher the recall (TPR), the more false positives
(FPR) the classifier produces. The dotted line represents the ROC curve of a purely
random classifier; a good classifier stays as far away from that line as possible (toward
the top-left corner)

One way to compare classifiers is to measure the area under the curve (AUC). A per‐ fect classifier will have a ROC 
AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. Scikit-Learn provides a function 
to compute the ROC AUC: 

"""

from sklearn.metrics import roc_auc_score

Auc = roc_auc_score(y_train_5, y_scores)

print(".."*10)
print(" The Area under the curve: ")
print(Auc)

"""
Since the ROC curve is so similar to the precision/recall (or PR)
curve, you may wonder how to decide which one to use. As a rule
of thumb, you should prefer the PR curve whenever the positive
class is rare or when you care more about the false positives than
the false negatives, and the ROC curve otherwise.
For example,
looking at the previous ROC curve (and the ROC AUC score), you
may think that the classifier is really good. But this is mostly
because there are few positives (5s) compared to the negatives
(non-5s). In contrast, the PR curve makes it clear that the classifier
has room for improvement (the curve could be closer to the top-right corner).
"""
