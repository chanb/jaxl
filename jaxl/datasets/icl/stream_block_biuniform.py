import numpy as np
import tensorflow as tf

from jaxl.datasets.icl.utils import TFDataset


class StreamBlockBiUniform:
    def __init__(
        self,
        num_high_prob_classes: int,
        num_low_prob_classes: int,
        high_prob: float,
        num_dims: int,
        seed: int,
    ):
        assert 0.0 < high_prob < 1.0
        assert (
            high_prob / num_high_prob_classes >= (1 - high_prob) / num_low_prob_classes
        )
        self.num_high_prob_classes = num_high_prob_classes
        self.num_low_prob_classes = num_low_prob_classes
        self.num_classes = num_high_prob_classes + num_low_prob_classes
        self.high_prob = high_prob
        self.low_prob = 1 - high_prob
        self.num_dims = num_dims
        self.rng = np.random.RandomState(seed)

        self.centers = self.rng.standard_normal(size=(self.num_classes, self.num_dims))
        self.centers /= np.linalg.norm(self.centers, axis=-1, keepdims=True)

    def get_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        fixed_start_pos: int = -1,
        abstract_class: int = 0,
        iid_context: int = 0,
        sample_low_prob_class_only: int = 0,
        sample_high_prob_class_only: int = 0,
        stratified: int = 0,
    ):
        assert sample_low_prob_class_only + sample_high_prob_class_only <= 1

        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        if iid_context:
            # Only do IID context
            while True:
                labels = self.rng.choice(
                    self.num_classes,
                    size=(num_examples + 1,),
                    p=weights,
                )

                if sample_low_prob_class_only:
                    labels[-1] = (
                        self.rng.choice(
                            self.num_low_prob_classes,
                            size=(1,),
                        )
                        + self.num_high_prob_classes
                    )

                inputs = self.centers[labels]
                inputs += input_noise_std * self.rng.randn(*inputs.shape)

                if abstract_class:
                    # Class 0 if high-prob lusters, class 1 otherwise
                    # TODO: Maybe there can be an ablation on varying number of classes?
                    labels = [
                        int(label < self.num_high_prob_classes) for label in labels
                    ]
                    labels = np.eye(2)[labels]
                else:
                    labels = np.eye(self.num_classes)[labels]

                yield {
                    "example": inputs,
                    "label": labels,
                }

        start_pos = fixed_start_pos
        available_low_prob_classes = list(
            range(
                self.num_high_prob_classes,
                self.num_classes,
            )
        )
        while True:
            if fixed_start_pos == -1:
                start_pos = self.rng.choice(num_examples)

            block_labels = self.rng.choice(
                self.num_classes,
                size=(2,),
                p=weights,
            )

            if stratified:
                # Stratified sampling
                # Choose low prob. class as query and removes it from being sampled onwards
                if (
                    len(available_low_prob_classes)
                    and self.rng.rand() >= self.high_prob
                ):
                    available_low_prob_idx = self.rng.randint(
                        len(available_low_prob_classes), size=(1,)
                    )
                    query_label = available_low_prob_classes[available_low_prob_idx]
                    available_low_prob_classes.pop(available_low_prob_idx)
                else:
                    query_label = self.rng.choice(
                        self.num_high_prob_classes,
                        size=(1,),
                    )
                block_labels[-1] = query_label
            elif sample_low_prob_class_only:
                # Sample low prob. class as query only
                block_labels[-1] = (
                    self.rng.choice(
                        self.num_low_prob_classes,
                        size=(1,),
                    )
                    + self.num_high_prob_classes
                )
            elif sample_high_prob_class_only:
                # Sample low prob. class as query only
                block_labels[-1] = (
                    self.rng.choice(
                        self.num_high_prob_classes,
                        size=(1,),
                    )
                )

            labels = [block_labels[0]] * (num_examples - start_pos) + [
                block_labels[1]
            ] * (start_pos + 1)

            inputs = self.centers[labels]
            inputs += input_noise_std * self.rng.randn(*inputs.shape)

            if abstract_class:
                # Class 0 if high-prob lusters, class 1 otherwise
                # TODO: Maybe there can be an ablation on varying number of classes?
                labels = [int(label < self.num_high_prob_classes) for label in labels]
                labels = np.eye(2)[labels]
            else:
                labels = np.eye(self.num_classes)[labels]

            yield {
                "example": inputs,
                "label": labels,
            }


def get_dataset(
    # For get_sequences
    num_examples: int,
    input_noise_std: float,
    fixed_start_pos: int = -1,
    abstract_class: int = 0,
    iid_context: int = 0,
    sample_low_prob_class_only: int = 0,
    sample_high_prob_class_only: int = 0,
    stratified: int = 0,
    # For constructor
    num_high_prob_classes: int = 16,
    num_low_prob_classes: int = 256,
    high_prob: float = 0.8,
    num_dims: int = 64,
    seed: int = 42,
):
    if abstract_class:
        num_classes = 2
    else:
        num_classes = num_low_prob_classes + num_high_prob_classes
    task = StreamBlockBiUniform(
        num_high_prob_classes,
        num_low_prob_classes,
        high_prob,
        num_dims,
        seed,
    )
    dataset = tf.data.Dataset.from_generator(
        task.get_sequences,
        args=(
            num_examples,
            input_noise_std,
            fixed_start_pos,
            abstract_class,
            iid_context,
            sample_low_prob_class_only,
            sample_high_prob_class_only,
            stratified,
        ),
        output_signature={
            "example": tf.TensorSpec(
                shape=(num_examples + 1, num_dims), dtype=tf.dtypes.float32
            ),
            "label": tf.TensorSpec(
                shape=(num_examples + 1, num_classes), dtype=tf.dtypes.int32
            ),
        },
    )
    return TFDataset(
        dataset,
        num_classes,
        (num_dims,),
        num_examples + 1,
    )
