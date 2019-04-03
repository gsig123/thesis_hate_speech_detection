import matplotlib.pyplot as plt


def plot_train_val_loss(history, save_to_file_path=None):
    """
    input: history which is the output from
           keras model.fit
    """
    plt.clf()
    history_dict = history.history
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save_to_file_path:
        plt.savefig(save_to_file_path)
    else:
        plt.show()


def plot_train_val_accuracy(history, save_to_file_path=None):
    """
    input: history which is the output from
           keras model.fit
    """
    plt.clf()
    history_dict = history.history
    train_metric = history_dict['acc']
    train_metric = history_dict['acc']
    val_metric = history_dict['val_' + 'acc']
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, 'r', label='Training {}'.format('acc'))
    plt.plot(epochs, val_metric, 'b', label='Validation {}'.format('acc'))
    plt.title('Training and Validation {}'.format('acc'))
    plt.xlabel('Epochs')
    plt.ylabel('{}'.format('acc'))
    plt.legend()
    if save_to_file_path:
        plt.savefig(save_to_file_path)
    else:
        plt.show()
