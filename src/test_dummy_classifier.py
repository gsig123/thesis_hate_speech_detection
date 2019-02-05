import numpy as np

from classifiers.classifier_dummy import DummyClassifier

c = DummyClassifier()
X = np.random.randint(100, size=100)
y_true = np.random.choice([0, 1], size=100)
y_pred = c.predict(X)
confusion_matrix = c.confusion_matrix(y_true, y_pred)
f1_score = c.f1_score(y_true, y_pred)
recall = c.recall(y_true, y_pred)
precision = c.precision(y_true, y_pred)
accuracy = c.accuracy(y_true, y_pred)

c.plot_confusion_matrix(confusion_matrix, class_names=["NOT", "OFF"], title="Dummy Classifier Not Normalized")
c.plot_confusion_matrix(confusion_matrix, class_names=["NOT", "OFF"], normalize=True, title="Dummy Classifier Normalized")


# Generate random y_true: 
# y_true = np.random.choice([0, 1], size=number_of_elements)

# Generate all false y_pred: 
# y_pred = np.zeros(number_of_elements,)
