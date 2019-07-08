from Pruner.FilterPruner import FilterPruner


class CompleteFilterPruner(FilterPruner):
    def __init__(self,
                 model,
                 sample_run,
                 force_forward_view=False,
                 ignore_last_conv=False):
        super(CompleteFilterPruner, self).__init__(model,
                                                   sample_run,
                                                   force_forward_view,
                                                   ignore_last_conv)

    def post_pruning_plan(self, filters_to_prune_per_layer):
        to_display = {}
        for index, curr_set in enumerate(self.sets):
            if len(curr_set) <= 1:
                continue

            set_as_list = list(curr_set)
            intersect_set = None
            for i, elem in enumerate(set_as_list):
                if elem not in filters_to_prune_per_layer:
                    intersect_set = set()
                elif i == 0:
                    intersect_set = set(filters_to_prune_per_layer[elem])
                else:
                    intersect_set = intersect_set.intersection(set(filters_to_prune_per_layer[elem]))

            if intersect_set is None:
                to_display[index] = "None"
            else:
                to_display[index] = len(intersect_set)

            if intersect_set is None or len(intersect_set) == 0:
                for elem in set_as_list:
                    if elem in filters_to_prune_per_layer:
                        del filters_to_prune_per_layer[elem]
            else:
                for elem in set_as_list:
                    filters_to_prune_per_layer[elem] = list(intersect_set)

        print("junction pruning size:", to_display)
