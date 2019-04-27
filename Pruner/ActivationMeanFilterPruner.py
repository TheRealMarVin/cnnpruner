from heapq import nsmallest
from operator import itemgetter

from Pruner.FilterPruner import FilterPruner


class ActivationMeanFilterPruner(FilterPruner):

    def __init__(self, model, sample_run):
        super(ActivationMeanFilterPruner, self).__init__(model, sample_run)

    def sort_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))
