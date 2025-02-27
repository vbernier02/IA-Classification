import pandas as pd
import numpy as np
import time

data = pd.read_csv("zoo.data", header=None)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

def distance(point1, point2):
    
    return np.sqrt(np.sum((point1 - point2) ** 2))

def precision(y_true, y_pred):
    
    correct_predictions = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct_predictions / total

def kppv(X_train, y_train, k_v, x_test):
    
    distances = [(distance(x, x_test), label) for x, label in zip(X_train, y_train)]
    distances.sort()
    k_nearest_labels = [label for _, label in distances[:k_v]]
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)

    return unique_labels[np.argmax(counts)]

k_fold = 5
indices = np.arange(len(X))
np.random.shuffle(indices)
fold_indices = np.array_split(indices, k_fold)
precisions = []
timers = []

for i in range(k_fold):
    test_indices = fold_indices[i]
    train_indices = np.concatenate([fold_indices[j] for j in range(k_fold) if j != i])
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    start_timer = time.time()
    k_v = 3
    y_pred = np.array([kppv(X_train, y_train, k_v, x_test) for x_test in X_test])
    end_timer = time.time()
    execution_time = end_timer - start_timer
    timers.append(execution_time)
    k_v = 3
    y_pred = np.array([kppv(X_train, y_train, k_v, x_test) for x_test in X_test])
    precision_val = precision(y_test, y_pred)
    precisions.append(precision_val)

timer_moyen = np.mean(timers)
meilleur_timer = np.min(timers)
pire_timer = np.max(timers)
precision_moyenne = np.mean(precisions)
meilleure_precision = np.max(precisions)
pire_precision = np.min(precisions)

print(f"Temps d'exécution moyen    : {timer_moyen} s")
print(f"Meilleur temps d'exécution : {meilleur_timer} s")
print(f"Pire temps d'exécution     : {pire_timer} s\n")
print(f"Précision moyenne          : {precision_moyenne}")
print(f"Meilleure précision        : {meilleure_precision}")
print(f"Pire précision             : {pire_precision}")
