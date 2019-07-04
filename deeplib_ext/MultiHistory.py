import matplotlib.pyplot as plt

from deeplib_ext.history import History


class MultiHistory:
    def __init__(self):
        self.data = {}

    def append_history(self, key, history):
        if key not in self.data:
            self.data[key] = history
        else:
            self.data[key] = self.data[key].append(history)

    def display_single_key(self, key, title="Training accuracy", xlabel="Epochs"):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(key)
        for k, v in self.data.items():
            epoch = len(v.history[History.TRAIN_ACC_KEY])
            epochs = [x for x in range(1, epoch + 1)]
            plt.plot(epochs, v.history[key], label=k)
        plt.legend()
        plt.show()
