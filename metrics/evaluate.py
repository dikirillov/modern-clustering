from sklearn import metrics
from accuracy import calc_accuracy


def evaluate(y_true, predictions, num_clusters):
    acc = calc_accuracy(y_true, predictions, num_clusters)
    nmi = metrics.normalized_mutual_info_score(y_true, predictions)
    ari = metrics.adjusted_rand_score(y_true, predictions)
    fmi = metrics.fowlkes_mallows_score(y_true, predictions)
    bcubed = BCubed(y_true)
    bc = bcubed(preds)
    return acc, nmi, ari, fmi, bc

