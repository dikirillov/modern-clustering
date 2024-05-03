from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from munkres import Munkres

def get_preds_labels(y_true, y_preds, num_clusters):
    conf_matrix = confusion_matrix(y_true, y_preds, labels=None)
    cost_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        n_cur_instances = cost_matrix[:, i].sum()
        cost_matrix[:, i] = n_cur_instances - conf_matrix[:, i]

    munkres_indicies = Munkres().compute(cost_matrix)
    y_preds_munkres = np.zeros_like(y_preds)
    permutated_clusters = np.zeros(num_clusters)
    for i in range(num_clusters):
        permutated_clusters[i] = int(munkres_indicies[i][1])

    return permutated_clusters[y_preds]

def calc_accuracy(y_true, y_preds, num_clusters):
    reordered_preds = get_preds_labels(y_true, y_preds, num_clusters)
    return accuracy_score(y_true, reordered_preds)
    
