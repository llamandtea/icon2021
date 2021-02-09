from sklearn import tree, model_selection
import pandas as pd
from sys import argv
from os import path


if not path.is_file(arv[1]):
    print("Error: could not find specified CSV dataset")

training_data = pd.read_csv(argv[1])
training_data = pd.get_dummies(training_data)
training_data_arr = training_data.to_numpy()

classifier = tree.DecisionTreeClassifier(max_depth=4)
classifier = classifier.fit(training_data_arr[:, 1:], training_data_arr[:, 0])

"""
dot_data = tree.export_graphviz(classifier, out_file="graph.dot",
                                feature_names=training_data.columns.to_list()[1:],
                                class_names=["True", "False"],
                                filled=True,
                                rounded=True)
"""

kf = model_selection.KFold(n_splits=10)
i = 0
j = 1
for train, test in kf.split(training_data_arr):
    classifier2 = tree.DecisionTreeClassifier(max_depth=4)
    classifier2.fit(training_data_arr[train, 1:], training_data_arr[train, 0])
    score = classifier2.score(training_data_arr[test, 1:], training_data_arr[test, 0])
    print("Score " + str(j) + ": " + str(score))
    j += 1
    i += score

print("Accuracy: " + str(i / 10))
