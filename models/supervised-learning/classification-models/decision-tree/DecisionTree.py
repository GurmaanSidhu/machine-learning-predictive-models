import pandas as pd 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

iris_data = load_iris()
iris = pd.DataFrame(
    np.c_[iris_data['data'], iris_data['target']],
    columns = iris_data['feature_names'] + ['species']
)

# split dataset
np.random.seed(678)
s = np.random.choice(iris.index, 100, replace=False)
iris_train = iris.loc[s]
iris_test = iris.drop(s)

x_train = iris_train.drop(columns=['species'])
y_train = iris_train['species']

x_test = iris_test.drop(columns=['species'])
y_test = iris_test['species']

# Train model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# plot tree
plot_tree(model, filled=True)
plt.show()

# predict and evaluate
y_pred = model.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Confusion Matrix:\n", conf_mat)
print(f"Model Accuracy: {accuracy:.10f}%")