
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

# Load MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data(path="mnist.npz")

# Reshape the input features
x_train = np.reshape(x_train, (60000, 28*28))

# Split the dataset into train and test sets
train_size = 0.7
random_state = 40
x_train, x_test, y_train, y_test = train_test_split(x_train[:6000], y_train[:6000], train_size=train_size, random_state=random_state)

# Initialize the normalizer
normalizer = Normalizer()

# Transform the features using the normalizer
x_train_norm  = normalizer.fit_transform(x_train)
x_test_norm = normalizer.transform(x_test)

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # Fit the model
    model.fit(features_train, target_train)

    # Make predictions on the testing features
    predictions = model.predict(features_test)

    # Calculate accuracy
    accuracy = (predictions == target_test).mean()

    # Return the accuracy
    return accuracy

# Initialize K-nearest neighbors classifier and perform grid search
knn = KNeighborsClassifier()
knn_clf = GridSearchCV(knn, dict(n_neighbors=[3,4], weights=['uniform', 'distance'], algorithm=['auto', 'brute']), scoring='accuracy', n_jobs=-1)
knn_clf.fit(x_train_norm, y_train)

# Initialize Random Forest classifier and perform grid search
rfc = RandomForestClassifier(random_state=40)
rfc_clf = GridSearchCV(rfc, dict(n_estimators=[300, 500], max_features=['auto', 'log2'], class_weight=['balanced', 'balanced_subsample']), scoring='accuracy', n_jobs=-1)
rfc_clf.fit(x_train_norm, y_train)

# Print K-nearest neighbors algorithm results
print("K-nearest neighbors algorithm")
print(f"Best estimator: {knn_clf.best_estimator_}")
print(f"Accuracy: {fit_predict_eval(knn_clf, x_train_norm, x_test_norm, y_train, y_test)}\n")

# Print Random Forest algorithm results
print("Random Forest algorithm")
print(f"Best estimator: {rfc_clf.best_estimator_}")
print(f"Accuracy: {fit_predict_eval(rfc_clf, x_train_norm, x_test_norm, y_train, y_test)}\n")
