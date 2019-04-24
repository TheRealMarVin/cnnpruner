import matplotlib.pyplot as plt
from IPython.display import clear_output


class History:

    def __init__(self):
        self.TRAIN_ACC_KEY = "train_acc"
        self.VAL_ACC_KEY = "val_acc"
        self.TRAIN_LOSS = "train_loss"
        self.VAL_LOSS = "val_loss"
        self.LR = "lr"
        self.STEP_ID = "step"

        self.history = {
            self.TRAIN_ACC_KEY: [],
            self.VAL_ACC_KEY: [],
            self.TRAIN_LOSS: [],
            self.VAL_LOSS: [],
            self.LR: [],
            self.STEP: []
        }

    def save(self, train_acc, val_acc, train_loss, val_loss, lr):
        self.history[self.TRAIN_ACC_KEY].append(train_acc)
        self.history[self.VAL_ACC_KEY].append(val_acc)
        self.history[self.TRAIN_LOSS].append(train_loss)
        self.history[self.VAL_LOSS].append(val_loss)
        self.history[self.LR].append(lr)

    def append(self, history):
        self.STEP.append(len(self.history[self.TRAIN_ACC_KEY]))
        self.history[self.TRAIN_ACC_KEY].extend(history[self.TRAIN_ACC_KEY])
        self.history[self.VAL_ACC_KEY].append(history[self.VAL_ACC_KEY])
        self.history[self.TRAIN_LOSS].append(history[self.TRAIN_LOSS])
        self.history[self.VAL_LOSS].append(history[self.VAL_LOSS])
        self.history[self.LR].append(history[self.LR])

    def display_accuracy(self):
        epoch = len(self.history[self.TRAIN_ACC_KEY])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(epochs, self.history[self.TRAIN_ACC_KEY], label='Train')
        plt.plot(epochs, self.history[self.VAL_ACC_KEY], label='Validation')
        plt.legend()
        plt.show()

    def display_loss(self):
        epoch = len(self.history[self.TRAIN_ACC_KEY])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(epochs, self.history[self.TRAIN_LOSS], label='Train')
        plt.plot(epochs, self.history[self.VAL_LOSS], label='Validation')
        plt.legend()
        plt.show()

    def display_lr(self):
        epoch = len(self.history[self.TRAIN_ACC_KEY])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel(self.LR)
        plt.plot(epochs, self.history[self.LR], label=self.LR)
        plt.show()

    def display(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]

        fig, axes = plt.subplots(3, 1)
        plt.tight_layout()

        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history[self.TRAIN_ACC_KEY], label='Train')
        axes[0].plot(epochs, self.history[self.VAL_ACC_KEY], label='Validation')
        axes[0].legend()

        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history[self.TRAIN_LOSS], label='Train')
        axes[1].plot(epochs, self.history[self.VAL_LOSS], label='Validation')

        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel(self.LR)
        axes[2].plot(epochs, self.history[self.LR], label=self.LR)

        plt.show()