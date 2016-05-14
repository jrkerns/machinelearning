from sklearn import datasets, model_selection, svm, neural_network, metrics

iris = datasets.load_digits()

iris_train, iris_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, train_size=0.7)

# parameters = {
#     'kernel': ['linear', 'rbf', 'sigmoid'], 'C': [1, 10, 100],
#     'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
# }
# classifier = model_selection.GridSearchCV(svm.SVC(), parameters)
parameters = {
    'hidden_layer_sizes': [(100, 20, 10), (50, 50)],
    'activation': ['logistic'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['invscaling']
}
classifier = model_selection.GridSearchCV(neural_network.MLPClassifier(), parameters)
classifier.fit(iris_train, y_train)

for params, mean_score, scores in classifier.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()
print(classifier.best_estimator_)
print("Best parameters found:")
print(classifier.best_params_)
print("With a training score of:")
print(classifier.best_score_)
print(metrics.f1_score(y_train, classifier.predict(iris_train), average='weighted'))
print("And test score of")
print(metrics.f1_score(y_test, classifier.predict(iris_test), average='weighted'))

