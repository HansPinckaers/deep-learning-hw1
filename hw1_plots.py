"""Plotting functions for the HW result files"""
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools

from matplotlib.axes import Axes

DISPLAY_LABELS = {
    'train': 'Train',
    'test': 'Test',
    'acc': 'Accuracy (%)',
    'loss': 'Loss (n.u.)'
}

DISPLAY_YLIMS = {
    'acc': (0, 100),
    'loss': (0, 2.5)
}

FIG_WIDTH = 12
FIG_HEIGHT = 9


def load_data(output_files):
    data = []
    for output_file in output_files:
        with open(output_file, mode='r', encoding='utf-8') as f:
            data += json.load(f)

    def sort_key(entry):
        return entry['model'], entry['optimizer']['name']

    data.sort(key=sort_key)

    return data


def get_model_results(data, model_name):
    for d in data:
        if d['model'].lower() == model_name.lower():
            yield d


def plot_model_results(model_results):
    model_results = list(model_results)
    model_name = model_results[0]['model']

    fig = plt.figure()

    prod = itertools.product(('train', 'test'), ('acc', 'loss'))
    for i, (train_test, acc_loss) in enumerate(prod):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_single_case(ax, train_test, acc_loss, model_results)

    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT, forward=True)
    fig.suptitle(model_name)
    fig.tight_layout(pad=2.5)

    return fig


def plot_single_case(ax: Axes, train_test, acc_loss, model_results):
    data_type = f'{train_test}_{acc_loss}'
    for curr_result in model_results:
        optimizer_name = curr_result['optimizer']['name']
        optimizer_params = curr_result['optimizer']['params']

        if optimizer_name == 'SGD':
            optimizer_label = f"SGD(lr={optimizer_params['lr']}, " \
                              f"mtm={optimizer_params['momentum']})"
        else:
            optimizer_label = f"{optimizer_name}(lr={optimizer_params['lr']})"

        data = np.array(curr_result[data_type])
        epochs = data[:, 0]
        values = data[:, 1]

        ax.plot(epochs, values, label=optimizer_label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(
            f"{DISPLAY_LABELS[train_test]} {DISPLAY_LABELS[acc_loss]}")
        ax.set_ylim(DISPLAY_YLIMS[acc_loss])
        ax.legend()
        ax.grid(b=True, which='both')


def plot_finetune_results(model_results):
    model_results = list(model_results)
    model_name = model_results[0]['model']
    fig = plt.figure()

    for i, acc_loss in enumerate(['acc', 'loss']):
        ax = fig.add_subplot(1, 2, i + 1)
        plot_single_case_finetune(ax, 'train', acc_loss, model_results)
        plot_single_case_finetune(ax, 'test', acc_loss, model_results)

    fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT, forward=True)
    fig.suptitle(f"{model_name} - Finetune SGD with momentum")
    fig.tight_layout(pad=2.5)

    return fig


def plot_single_case_finetune(ax: Axes, train_test, acc_loss, model_results):
    data_type = f'{train_test}_{acc_loss}'
    for curr_result in model_results:
        data = np.array(curr_result[data_type])
        epochs = data[:, 0]
        values = data[:, 1]

        ax.plot(epochs, values, label=train_test)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{DISPLAY_LABELS[acc_loss]}")
        ax.set_ylim(auto=True)
        ax.legend()
        ax.grid(b=True, which='both')
