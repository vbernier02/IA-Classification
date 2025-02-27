import pandas as pd
import numpy as np
import time

data = pd.read_csv("zoo.data", header=None)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

def naive_bayes(X_train, y_train):
    unique_labels, counts = np.unique(y_train, return_counts=True)
    class_proba = counts / len(y_train)
    
    feature_proba = []
    for label in unique_labels:
        label_mask = (y_train == label)
        label_counts = np.sum(X_train[label_mask, :], axis=0) + 1 
        label_probas = label_counts / np.sum(label_counts)
        feature_proba.append(label_probas)
    
    return unique_labels, class_proba, np.array(feature_proba)

def predict_naive_bayes(unique_labels, class_proba, feature_proba, x_test):
    log_probas = np.log(class_proba)
    for i, label in enumerate(unique_labels):
        log_probas[i] += np.sum(np.log(feature_proba[i, :]) * x_test)
    return unique_labels[np.argmax(log_probas)]

def precision(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

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
    unique_labels, class_proba, feature_proba = naive_bayes(X_train, y_train)
    y_pred = np.array([predict_naive_bayes(unique_labels, class_proba, feature_proba, x_test) for x_test in X_test])
    end_timer = time.time()
    execution_time = end_timer - start_timer
    timers.append(execution_time)      
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
