import numpy as np
import sys
sys.path.append("..")
from classifiers.classifier_dummy import DummyClassifier

c = DummyClassifier()
X = np.random.randint(100, size=100)
y_true = np.random.choice([0, 1], size=100)
y_pred = c.predict(X)
f1_score = c.f1_score(y_true, y_pred)
recall = c.recall(y_true, y_pred)
precision = c.precision(y_true, y_pred)
accuracy = c.accuracy(y_true, y_pred)

