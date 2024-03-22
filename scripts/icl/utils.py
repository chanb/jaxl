from jaxl.constants import *
from jaxl.datasets import get_dataset
from jaxl.plot_utils import set_size

import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


# Plot dataset example
def plot_examples(
    dataset, num_examples, save_path, exp_name, eval_name, doc_width_pt=500
):
    nrows = num_examples
    ncols = dataset._dataset.sequence_length

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=set_size(doc_width_pt, 0.95, (nrows, ncols), False),
        layout="constrained",
    )

    for example_i in range(num_examples):
        ci, co, q, l = dataset[example_i]

        for idx, (img, label) in enumerate(zip(ci, co)):
            axes[example_i, idx].imshow(img)
            axes[example_i, idx].set_title(np.argmax(label))
            axes[example_i, idx].axis("off")
        axes[example_i, -1].axis("off")
        axes[example_i, -1].imshow(q[0])
        axes[example_i, -1].set_title(np.argmax(l, axis=-1))

    fig.savefig(
        os.path.join(save_path, "plots", exp_name, "examples-{}.pdf".format(eval_name)),
        format="pdf",
        bbox_inches="tight",
        dpi=600,
    )


# Get model predictions
def get_preds_labels(model, params, data_loader, num_tasks, max_label=None):
    all_preds = []
    all_labels = []
    all_outputs = []
    num_query_class_in_context = []

    for batch_i, samples in enumerate(data_loader):
        if batch_i >= num_tasks:
            break

        (context_inputs, context_outputs, queries, one_hot_labels) = samples

        outputs, _, _ = model.forward(
            params[CONST_MODEL_DICT][CONST_MODEL],
            queries.numpy(),
            {
                CONST_CONTEXT_INPUT: context_inputs.numpy(),
                CONST_CONTEXT_OUTPUT: context_outputs.numpy(),
            },
            eval=True,
        )
        # return train_outputs, train_updates, outputs, updates
        if max_label is None:
            preds = np.argmax(outputs, axis=-1)
        elif max_label == CONST_AUTO:
            print(data_loader.dataset._data["num_classes"])
            preds = np.argmax(
                outputs[..., : data_loader.dataset._data["num_classes"]], axis=-1
            )
        else:
            preds = np.argmax(outputs[..., :max_label], axis=-1)
        labels = np.argmax(one_hot_labels.numpy(), axis=-1)
        all_preds.append(preds)
        all_labels.append(labels)
        all_outputs.append(outputs)
        num_query_class_in_context.append(
            np.max(
                np.argmax(context_outputs.numpy(), axis=-1) == labels[:, None], axis=-1
            )
        )

    all_outputs = np.concatenate(all_outputs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    num_query_class_in_context = np.concatenate(num_query_class_in_context)
    return all_preds, all_labels, all_outputs, num_query_class_in_context


# Check model accuracy
def print_performance_with_aux(
    all_preds,
    all_labels,
    num_query_class_in_context,
    output_dim,
    context_len,
    fixed_length=True,
):
    result_str = ""
    conf_mat = confusion_matrix(all_labels, all_preds, labels=np.arange(output_dim))
    auxes = {}
    acc = np.trace(conf_mat) / np.sum(conf_mat) * 100
    result_str += "Accuracy: {}% - ".format(acc)
    result_str += "Query class in context ratio: {}\n".format(
        np.mean(num_query_class_in_context)
    )
    auxes["all"] = {
        "accuracy": acc,
        "query_class_in_context_ratio": np.mean(num_query_class_in_context),
    }

    if not fixed_length:
        all_labels = all_labels.reshape((-1, context_len))
        all_preds = all_preds.reshape((-1, context_len))
        num_query_class_in_context = num_query_class_in_context.reshape(
            (-1, context_len)
        )

        for len_i in range(context_len):
            conf_mat = confusion_matrix(
                all_labels[:, len_i], all_preds[:, len_i], labels=np.arange(output_dim)
            )
            acc = np.trace(conf_mat) / np.sum(conf_mat) * 100
            result_str += "Context Length: {} - Accuracy: {}% - ".format(len_i + 1, acc)
            result_str += "Query class in context ratio: {}\n".format(
                np.mean(num_query_class_in_context[:, len_i])
            )
            auxes[len_i + 1] = {
                "accuracy": acc,
                "query_class_in_context_ratio": np.mean(
                    num_query_class_in_context[:, len_i]
                ),
            }

    return auxes, result_str


# Get dataloader
def get_data_loader(
    dataset_config,
    seed,
    batch_size,
    num_workers,
    visualize=False,
):
    dataset = get_dataset(
        dataset_config,
        seed,
    )

    if visualize:
        plot_examples(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataset, data_loader


# Complete evaluation
def evaluate(
    model,
    params,
    dataset,
    data_loader,
    num_tasks,
    max_label,
    context_len,
    fixed_length=True,
):
    preds, labels, outputs, num_query_class_in_context = get_preds_labels(
        model, params, data_loader, num_tasks, max_label
    )
    auxes, _ = print_performance_with_aux(
        preds,
        labels,
        num_query_class_in_context,
        dataset.output_dim[0],
        context_len,
        fixed_length,
    )
    return auxes["all"]["accuracy"], auxes
