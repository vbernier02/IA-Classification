import pandas as pd
import numpy as np
import time

data = pd.read_csv("zoo.data", header=None)
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

class TreeNode:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.feature_index = None
        self.limites = None
        self.left = None
        self.right = None
        self.output = None

def precision(y_true, y_pred):
    
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

def gini(labels):
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini_value = 1 - np.sum(probabilities ** 2)
    return gini_value

def split_data(data, labels, feature_index, limite):
    
    left_mask = data[:, feature_index] <= limite
    right_mask = ~left_mask
    left_data, left_labels = data[left_mask], labels[left_mask]
    right_data, right_labels = data[right_mask], labels[right_mask]
    return left_data, left_labels, right_data, right_labels

def best_split(data, labels):
    
    m, n = data.shape
    initial_gini = gini(labels)
    best_gini = float('inf')
    best_feature_index = None
    best_limite = None

    for feature_index in range(n):
        limites = np.unique(data[:, feature_index])
        for limite in limites:
            left_data, left_labels, right_data, right_labels = split_data(data, labels, feature_index, limite)
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            left_gini = gini(left_labels)
            right_gini = gini(right_labels)
            weighted_gini = (len(left_labels) / m) * left_gini + (len(right_labels) / m) * right_gini

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature_index
                best_limite = limite

    return best_feature_index, best_limite

def build_tree(data, labels, depth=0, max_depth=None):
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    output = unique_labels[np.argmax(counts)]
    node = TreeNode(data, labels)
    node.output = output

    if depth == max_depth or len(np.unique(labels)) == 1:
        return node

    feature_index, limite = best_split(data, labels)

    if feature_index is not None:
        left_data, left_labels, right_data, right_labels = split_data(data, labels, feature_index, limite)
        node.feature_index = feature_index
        node.limites = limite
        node.left = build_tree(left_data, left_labels, depth + 1, max_depth)
        node.right = build_tree(right_data, right_labels, depth + 1, max_depth)

    return node

def predict_sample(tree, sample):
    
    if tree.output is not None:
        return tree.output

    if sample[tree.feature_index] <= tree.limites:
        return predict_sample(tree.left, sample)
    else:
        return predict_sample(tree.right, sample)

def predict(tree, data):
    
    predictions = [predict_sample(tree, sample) for sample in data]
    return np.array(predictions)

k_fold = 5
indices = np.arange(len(X))
np.random.shuffle(indices)
fold_indices = np.array_split(indices, k_fold)
precisions = []
timers = []

for i in range(k_fold):
    test_indices = fold_indices[i]
    train_indices = np.concatenate([fold_indices[j] for j in range(k_fold) if j != i])
    train_data, test_data = X[train_indices], X[test_indices]
    train_labels, test_labels = y[train_indices], y[test_indices]
    start_timer = time.time()
    tree = build_tree(train_data, train_labels, max_depth=None)
    end_timer = time.time()
    execution_time = end_timer - start_timer
    timers.append(execution_time)
    predictions = predict(tree, test_data)
    precision_val = precision(test_labels, predictions)
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
