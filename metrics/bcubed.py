import numpy as np


class BCubed:
    def __init__(self, gold_standard):
        self.gold_standard = gold_standard

    def correctness(self, label, index, labels, same_cluster_indexes):
        same_predictions = labels[same_cluster_indexes] == label
        same_real = self.gold_standard[same_cluster_indexes] == self.gold_standard[index]
        return (same_predictions * same_real).sum()

    def precition_bcubed(self, labels):
        precition_bcubed = 0
        for index, label in enumerate(labels):
            same_cluster_indexes = np.where(self.gold_standard == self.gold_standard[index])[0]
            precition_bcubed += self.correctness(label, index, labels, same_cluster_indexes) / len(same_cluster_indexes)
        return precition_bcubed / len(labels)


    def recall_bcubed(self, labels):
        recall_bcubed = 0
        for index, label in enumerate(labels):
            same_cluster_indexes = np.where(labels == label)[0]
            recall_bcubed += self.correctness(label, index, labels, same_cluster_indexes) / len(same_cluster_indexes)
        return recall_bcubed / len(labels)

    def __call__(self, labels):
        precision = self.precition_bcubed(labels)
        recall = self.recall_bcubed(labels)
        return 2 * precision * recall / (precision + recall)
