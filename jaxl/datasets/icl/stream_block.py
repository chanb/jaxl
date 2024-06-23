import numpy as np
import tensorflow as tf

from jaxl.datasets.icl.utils import TFDataset


class StreamBlock:
    def __init__(
        self,
        num_base_classes: int,
        num_clusters: int,
        num_abstract_classes: int,
        num_dims: int,
        seed: int,
        novel_abstract_class: bool = False,
    ):
        assert num_clusters <= num_base_classes
        self.num_base_classes = num_base_classes
        self.num_clusters = num_clusters
        self.num_abstract_classes = num_abstract_classes
        self.num_dims = num_dims
        self.rng = np.random.RandomState(seed)

        self.base_centers = self.rng.standard_normal(
            size=(self.num_base_classes, self.num_dims)
        )
        self.base_centers /= np.linalg.norm(self.base_centers, axis=-1, keepdims=True)
        self.generate_abstract_class_map(novel_abstract_class)

    def generate_abstract_class_map(self, novel_abstract_class):
        if novel_abstract_class:
            self.rng.random()

        # Randomly assign
        num_base_per_cluster = int(np.ceil(self.num_base_classes / self.num_clusters))
        base_to_cluster = self.rng.permutation(
            np.tile(np.arange(self.num_clusters), reps=(num_base_per_cluster,))[
                : self.num_base_classes
            ]
        )

        self.cluster_to_base_map = dict()
        for cluster_class in range(self.num_clusters):
            self.cluster_to_base_map[cluster_class] = np.where(
                base_to_cluster == cluster_class
            )[0]

    def get_random_contexts(self, num_examples: int):
        while True:
            inputs = self.rng.standard_normal(size=(num_examples + 1, self.num_dims))
            inputs /= np.linalg.norm(inputs, axis=-1, keepdims=True)
            labels = self.rng.randint(
                self.num_abstract_classes, size=(num_examples + 1,)
            )
            labels[-1] = labels[-2]
            labels = np.eye(self.num_abstract_classes)[labels]

            yield {
                "example": inputs,
                "label": labels,
            }

    def get_sequences(
        self,
        num_examples: int,
        zipf_exp: float,
        input_noise_std: float,
        fixed_start_pos: int = -1,
        prefix_first_clusters: int = 0,
        # The next two evals whether or not the last label impacts model prediction
        eval_perturb_last_label: int = 0,
        eval_random_context: int = 0,  # Overwrites the original iterator
    ):
        if eval_random_context:
            self.get_random_contexts(num_examples)

        # NOTE: The zipfian distribution skews towards smaller class labels.
        zipf_weights = np.array(
            [1 / j**zipf_exp for j in range(self.num_abstract_classes, 0, -1)]
        )
        zipf_weights /= np.sum(zipf_weights)

        def get_input(label):
            base_class = self.rng.choice(self.cluster_to_base_map[label])
            input = self.base_centers[base_class]
            input += input_noise_std * self.rng.randn(*input.shape)
            return input

        start_pos = fixed_start_pos
        while True:
            if fixed_start_pos == -1:
                start_pos = self.rng.choice(num_examples)

            clusters = self.rng.choice(self.num_clusters, size=(2,))

            # NOTE: Allow two blocks with same abstract class
            abstract_classes = self.rng.choice(
                self.num_abstract_classes,
                size=(2,),
                p=zipf_weights,
            )

            # NOTE: This is to have deterministic labels for specific clusters
            if clusters[0] < prefix_first_clusters:
                abstract_classes[0] = clusters[0] % self.num_abstract_classes
            if clusters[1] < prefix_first_clusters:
                abstract_classes[1] = clusters[1] % self.num_abstract_classes

            cluster_labels = [clusters[0]] * (num_examples - start_pos) + [
                clusters[1]
            ] * (start_pos + 1)
            labels = [abstract_classes[0]] * (num_examples - start_pos) + [
                abstract_classes[1]
            ] * (start_pos + 1)

            if eval_perturb_last_label:
                new_label = 0
                while new_label in abstract_classes:
                    new_label = self.rng.randint(self.num_abstract_classes)
                labels[-2:] = new_label

            labels = np.eye(self.num_abstract_classes)[labels]
            inputs = np.array(list(map(get_input, cluster_labels)))

            yield {
                "example": inputs,
                "label": labels,
            }


def get_dataset(
    # For get_sequences
    num_examples: int,
    zipf_exp: float,
    input_noise_std: float,
    fixed_start_pos: int = -1,
    eval_perturb_last_label: bool = False,
    eval_random_context: bool = False,
    # For constructor
    num_base_classes: int = 10,
    num_clusters: int = 10,
    num_abstract_classes: int = 2,
    num_dims: int = 64,
    seed: int = 42,
    novel_abstract_class: bool = False,
):
    task = StreamBlock(
        num_base_classes,
        num_clusters,
        num_abstract_classes,
        num_dims,
        seed,
        novel_abstract_class,
    )
    dataset = tf.data.Dataset.from_generator(
        task.get_sequences,
        args=(
            num_examples,
            zipf_exp,
            input_noise_std,
            fixed_start_pos,
            eval_perturb_last_label,
            eval_random_context,
        ),
        output_signature={
            "example": tf.TensorSpec(
                shape=(num_examples + 1, num_dims), dtype=tf.dtypes.float32
            ),
            "label": tf.TensorSpec(
                shape=(num_examples + 1, num_abstract_classes), dtype=tf.dtypes.int32
            ),
        },
    )
    return TFDataset(
        dataset,
        num_abstract_classes,
        (num_dims,),
        num_examples + 1,
    )