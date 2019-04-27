import random

from Pruner.FilterPruner import FilterPruner


class RandomFilterPruner(FilterPruner):

    def __init__(self, model, sample_run):
        super(RandomFilterPruner, self).__init__(model, sample_run)

    def sort_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return random.sample(data, num)
