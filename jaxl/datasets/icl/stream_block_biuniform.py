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
        linearly_separable: bool = False,
        flip_label: bool = False,
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

        if linearly_separable:
            boundary = self.rng.uniform(
                low=-1.0,
                high=1.0,
                size=(self.num_dims + 1, 1),
            )
            boundary[0] = 0.0  # Pass through origin
            margin = 0.2

            done_generation = False
            high_prob_centers = np.zeros((self.num_high_prob_classes, self.num_dims))
            replace_mask = high_prob_centers == 0

            while not done_generation:
                new_samples = self.rng.standard_normal(
                    size=(self.num_high_prob_classes, self.num_dims)
                )
                new_samples /= np.linalg.norm(new_samples, axis=-1, keepdims=True)

                high_prob_centers = (
                    high_prob_centers * (1 - replace_mask) + new_samples * replace_mask
                )
                dists = (high_prob_centers @ boundary[1:] + boundary[:1]) / np.sqrt(
                    np.sum(boundary[1:] ** 2)
                )
                replace_mask = dists < margin if flip_label else dists > -margin
                done_generation = np.sum(replace_mask) == 0
            print("Generated high prob centers")

            done_generation = False
            low_prob_centers = np.zeros((self.num_low_prob_classes, self.num_dims))
            replace_mask = low_prob_centers == 0
            while not done_generation:
                new_samples = self.rng.standard_normal(
                    size=(self.num_low_prob_classes, self.num_dims)
                )
                new_samples /= np.linalg.norm(new_samples, axis=-1, keepdims=True)

                low_prob_centers = (
                    low_prob_centers * (1 - replace_mask) + new_samples * replace_mask
                )
                dists = (low_prob_centers @ boundary[1:] + boundary[:1]) / np.sqrt(
                    np.sum(boundary[1:] ** 2)
                )
                replace_mask = dists > -margin if flip_label else dists < margin
                done_generation = np.sum(replace_mask) == 0
            print("Generated low prob centers")

            self.centers = np.concatenate((high_prob_centers, low_prob_centers), axis=0)
        else:
            self.centers = self.rng.standard_normal(
                size=(self.num_classes, self.num_dims)
            )
            self.centers /= np.linalg.norm(self.centers, axis=-1, keepdims=True)

    def get_iid_context_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        abstract_class: int = 0,
    ):
        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        # Only do IID context
        while True:
            labels = self.rng.choice(
                self.num_classes,
                size=(num_examples + 1,),
                p=weights,
            )

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

    def get_non_iid_stratified_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        fixed_start_pos: int = -1,
        abstract_class: int = 0,
        stratified: int = 0,
    ):
        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        start_pos = fixed_start_pos
        low_prob_classes_sample_counts = np.zeros(self.num_low_prob_classes)
        while True:
            if fixed_start_pos == -1:
                start_pos = self.rng.choice(num_examples)

            block_labels = self.rng.choice(
                self.num_classes,
                size=(2,),
                p=weights,
            )

            # Stratified sampling
            # Choose low prob. class as query and removes it from being sampled onwards
            available_low_prob_classes = np.where(
                low_prob_classes_sample_counts < stratified
            )[0]
            if len(available_low_prob_classes) and self.rng.rand() >= self.high_prob:
                query_label = self.rng.choice(available_low_prob_classes)
                low_prob_classes_sample_counts[query_label] += 1
                query_label += self.num_high_prob_classes
            else:
                query_label = self.rng.choice(
                    self.num_high_prob_classes,
                    size=(1,),
                )
            block_labels[-1] = query_label

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

    def get_sequences(
        self,
        num_examples: int,
        input_noise_std: float,
        fixed_start_pos: int = -1,
        abstract_class: int = 0,
        sample_low_prob_class_only: int = 0,
        sample_high_prob_class_only: int = 0,
    ):
        assert sample_low_prob_class_only + sample_high_prob_class_only <= 1

        # NOTE: The zipfian distribution skews towards smaller class labels.
        weights = [
            self.high_prob / self.num_high_prob_classes
        ] * self.num_high_prob_classes + [
            self.low_prob / self.num_low_prob_classes
        ] * self.num_low_prob_classes

        start_pos = fixed_start_pos
        while True:
            if fixed_start_pos == -1:
                start_pos = self.rng.choice(num_examples)

            block_labels = self.rng.choice(
                self.num_classes,
                size=(2,),
                p=weights,
            )

            if sample_low_prob_class_only:
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
                block_labels[-1] = self.rng.choice(
                    self.num_high_prob_classes,
                    size=(1,),
                )

            labels = [block_labels[0]] * (num_examples - start_pos) + [
                block_labels[1]
            ] * (start_pos + 1)

            inputs = self.centers[labels]
            inputs += input_noise_std * self.rng.randn(*inputs.shape)

            if abstract_class:
                # Class 0 if high-prob clusters, class 1 otherwise
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
    sample_low_prob_class_only: int = 0,
    sample_high_prob_class_only: int = 0,
    stratified: int = 0,
    # For constructor
    num_high_prob_classes: int = 16,
    num_low_prob_classes: int = 256,
    high_prob: float = 0.8,
    num_dims: int = 64,
    mode: str = "default",
    seed: int = 42,
    linearly_separable: bool = False,
    flip_label: bool = False,
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
        linearly_separable,
        flip_label,
    )

    if mode == "iid_context":
        seq_generator = task.get_iid_context_sequences
        args = (
            num_examples,
            input_noise_std,
            abstract_class,
        )
    elif mode == "non_iid_stratified":
        seq_generator = task.get_non_iid_stratified_sequences
        args = (
            num_examples,
            input_noise_std,
            fixed_start_pos,
            abstract_class,
            stratified,
        )
    elif mode == "default":
        seq_generator = task.get_sequences
        args = (
            num_examples,
            input_noise_std,
            fixed_start_pos,
            abstract_class,
            sample_low_prob_class_only,
            sample_high_prob_class_only,
        )
    else:
        raise NotImplementedError

    dataset = tf.data.Dataset.from_generator(
        seq_generator,
        args=args,
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
