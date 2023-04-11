#%%
from __future__ import print_function

from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, recall_score
from sklearn import svm

import seaborn as sns

from IPython.display import display


x, y = datasets.load_digits(return_X_y=True)



original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(x, y,
                                                    test_size=.5,
                                                    random_state=183212)

print('amostras em treino: %i' % original_x_train.shape[0],
      'amostras em teste: %i' % original_x_test.shape[0],
      'número de características: %i' % original_x_train.shape[1],
      'número de classes: %i' % (np.max(original_y_train) + 1),
      sep='\n', end='\n\n')


# %%
plt.figure(figsize=(16, 8))

for ix in range(8  * 32):
    plt.subplot(8, 32, ix + 1)
    plt.imshow(original_x_train[ix].reshape(8, 8), cmap='Greys')
    plt.axis('off')
    
    
# %%
from sklearn.manifold import TSNE

encoder2D = TSNE()
w_train = encoder2D.fit_transform(original_x_train)
w_test = encoder2D.fit_transform(original_x_test)

plt.figure(figsize=(16, 6))
categorical_colors = sns.color_palette()

for ix, (x, y) in enumerate(((w_train, original_y_train), (w_test, original_y_test))):
    plt.subplot(1, 2, ix + 1)
    sns.scatterplot(x=x[:,0].T, y=x[:,1].T, hue=y, palette=categorical_colors);

# %%
plt.figure(figsize=(16, 4))

plt.subplot(121)
plt.title('Frequencia das classes no conjunto de treinamento (%i amostras)' % len(original_x_train))
labels, counts = np.unique(original_y_train, return_counts=True)
sns.barplot(x=labels, y=counts)

plt.subplot(122)
plt.title('Frequencia das classes no conjunto de teste (%i amostras)' % len(original_x_test))
labels, counts = np.unique(original_y_test, return_counts=True)
sns.barplot(x=labels, y=counts);






# %%
from itertools import product
def train_test_plot_rbf(data, penalty_array=[1e-3, 1e-2, 1e-1, 1.0, 10.0], gamma_array= [1e-4, 1e-3, 1e-2, 1e-1], kernel='rbf'):
    x_train, x_test, y_train, y_test = data
    plt.figure(figsize=(24, 18))
    
    for (k, (penalty, gamma)) in enumerate(product(penalty_array, gamma_array)):
        clf = svm.SVC(kernel=kernel, C=penalty, gamma=gamma)
        clf.fit(x_train, y_train)

        predicted_train = clf.predict(x_train)
        predicted_test = clf.predict(x_test)
        

        # calculamos a acc de treino e teste
        train_acc = accuracy_score(y_train, predicted_train)
        test_acc = accuracy_score(y_test, predicted_test)
        
        print("Acurácia treino: ", train_acc, " - acurácia teste: ", test_acc, " - gamma: ", gamma, " - pennalty: ", penalty)
        


def get_model_and_prediction(data, penalty=1e-3, gamma=1e-1, kernel='rbf'):
    x_train, x_test, y_train, _ = data
    
    clf = svm.SVC(kernel=kernel, C=penalty, gamma=gamma)
    clf.fit(x_train, y_train)

    predicted_test = clf.predict(x_test)
        
    return (clf, predicted_test)

# %%
def show_predict_digits(predicted_test, num_test=5):
    
    _, axes = plt.subplots(nrows=1, ncols=num_test, figsize=(10, 3))
    for ax, image, prediction in zip(axes, original_x_test, predicted_test[0:num_test]):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def calc_model_metrics(model, y_test, predicted):
    print(
        f"Classification report for classifier {model}:\n"
        f"{classification_report(y_test, predicted)}\n"
    )
    print('Classificações corretas =', accuracy_score(original_y_test, best_prediction, normalize=False))
    print('Acurácia =', accuracy_score(original_y_test, best_prediction))
    print('Acurácia normalizada =', recall_score(original_y_test, best_prediction, average='macro'))

def plot_confusion_matrix(y_test, predicted):
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Matriz de confusão")
    print(f"Matriz de confusão:\n{disp.confusion_matrix}")

    plt.show()

# %%

trainTestData = []
trainTestData.append((original_x_train, original_x_test, original_y_train, original_y_test))

train_test_plot_rbf(trainTestData[0])

best_model, best_prediction = get_model_and_prediction(trainTestData[0], penalty=1.0, gamma=0.001)


# %%
show_predict_digits(best_prediction, num_test=5)
calc_model_metrics(best_model, original_y_test, best_prediction)

plot_confusion_matrix(original_y_test, best_prediction)


# %%
# Teste com as Random Forest
from sklearn.ensemble import RandomForestClassifier

rfModel = RandomForestClassifier(n_estimators=10)
rfModel.fit(original_x_train, original_y_train)

#Predizer os dados de teste
random_forest_prediction = rfModel.predict(original_x_test)

calc_model_metrics(rfModel, original_y_test, random_forest_prediction)

plot_confusion_matrix(original_y_test, random_forest_prediction)


# %%
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

lr = LogisticRegression(
            solver="saga",
            multi_class="ovr",
            penalty="l1",
            max_iter=3,
            random_state=42,
        )

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
lr.fit(original_x_train, original_y_train)
lr_prediction = lr.predict(original_x_test)

calc_model_metrics(lr, original_y_test, lr_prediction)

plot_confusion_matrix(original_y_test, lr_prediction)

# %%
