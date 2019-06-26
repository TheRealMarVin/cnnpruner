from Pruner.FilterPruner import FilterPruner


class CompleteFilterPruner(FilterPruner):
    def __init__(self, model, sample_run, force_forward_view=False):
        super(CompleteFilterPruner, self).__init__(model, sample_run, force_forward_view)

