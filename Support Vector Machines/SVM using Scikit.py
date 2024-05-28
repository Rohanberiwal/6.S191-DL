import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import hinge_loss

X, y = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='linear', C=1.0)  
svm_model.fit(X_train, y_train)
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
print(f'Weight vector (w): {w}')
print(f'Bias  (b): {b}')

y_pred = svm_model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

hinge_losses = hinge_loss(y_test, svm_model.decision_function(X_test))
print(f'Hinge Loss: {hinge_losses}')
