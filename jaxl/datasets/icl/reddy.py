import numpy as np
import tensorflow as tf

from jaxl.datasets.icl.utils import TFDataset


class Reddy:
    def __init__(
        self,
        num_base_classes: int,
        num_abstract_classes: int,
        num_dims: int,
        seed: int,
        base_per_abstract_map: str = None,
        novel_abstract_class: bool = False,
    ):
        """
        NOTE: For experiments:
        - IWL eval: set p_bursty=0.0
        - IWL eval with no context: set empty_examples = True in get_sequences method
        - ICL eval with novel input: use new seed with p_bursty=1 and bursty_len=4
        - ICL eval with permuted label: set novel_abstract_class = True with p_bursty=1 and bursty_len=4
        """
        assert num_abstract_classes < num_base_classes
        self.num_base_classes = num_base_classes
        self.num_abstract_classes = num_abstract_classes
        self.num_dims = num_dims
        self.rng = np.random.RandomState(seed)

        self.base_centers = self.rng.standard_normal(
            size=(self.num_base_classes, self.num_dims)
        )
        self.base_centers /= np.linalg.norm(self.base_centers, axis=-1, keepdims=True)
        self.generate_abstract_class_map(base_per_abstract_map, novel_abstract_class)

    def generate_abstract_class_map(self, base_per_abstract_map, novel_abstract_class):
        if novel_abstract_class:
            self.rng.random()

        if base_per_abstract_map == "l2":
            # Closest L2
            abstract_centers = self.rng.permutation(len(self.base_centers))[
                : self.num_abstract_classes
            ]
            abstract_centers = self.base_centers[abstract_centers]
            base_to_abstract = np.argmin(
                np.linalg.norm(
                    self.base_centers[:, None] - abstract_centers[None, :], axis=-1
                ),
                axis=1,
            )
        else:
            # Randomly assign
            num_base_per_abstract = int(
                np.ceil(self.num_base_classes / self.num_abstract_classes)
            )
            base_to_abstract = self.rng.permutation(
                np.tile(
                    np.arange(self.num_abstract_classes), reps=(num_base_per_abstract,)
                )[: self.num_base_classes]
            )

        self.abstract_to_base_map = dict()
        for abstract_class in range(self.num_abstract_classes):
            self.abstract_to_base_map[abstract_class] = np.where(
                base_to_abstract == abstract_class
            )[0]

    def get_sequences(
        self,
        num_examples: int,
        p_bursty: int,
        bursty_len: int,
        zipf_exp: float,
        input_noise_std: float,
        target_allowed_in_example: bool = False,
        empty_examples: bool = False,
    ):
        # NOTE: The zipfian distribution skews towards smaller class labels.
        zipf_weights = np.array(
            [1 / j**zipf_exp for j in range(self.num_abstract_classes, 0, -1)]
        )
        zipf_weights /= np.sum(zipf_weights)
        min_bursty_examples = bursty_len * 2
        generate_extra_distractors = num_examples > min_bursty_examples

        def get_input(label):
            base_class = self.rng.choice(self.abstract_to_base_map[label])
            input = self.base_centers[base_class]
            input += input_noise_std * self.rng.randn(*input.shape)
            return input

        while True:
            target = self.rng.choice(
                self.num_abstract_classes,
                p=zipf_weights,
            )

            if empty_examples:
                labels = np.array([target]).astype(int)
                inputs = np.array(list(map(get_input, labels)))

                labels = np.eye(self.num_abstract_classes)[labels]
                labels = np.concatenate(
                    (np.full((num_examples, *labels.shape[1:]), fill_value=-1), labels)
                )
                inputs = np.concatenate(
                    (np.zeros((num_examples, *inputs.shape[1:])), inputs),
                    axis=0,
                )
                yield {
                    "example": inputs,
                    "label": labels,
                }
                continue

            is_bursty = self.rng.uniform() <= p_bursty
            if is_bursty:
                # Ensure distractor label isn't the target
                done = False
                while not done:
                    distractor_label = self.rng.choice(self.num_abstract_classes)
                    done = target != distractor_label

                labels = []
                # Generate extra distractor labels that are not target nor distractor
                if generate_extra_distractors:
                    done = False
                    while not done:
                        labels = self.rng.choice(
                            self.num_abstract_classes,
                            size=(num_examples - min_bursty_examples),
                        )
                        done = target not in labels and distractor_label not in labels

                # Permute examples
                labels = self.rng.permutation(
                    np.concatenate(
                        [
                            [target] * bursty_len,
                            [distractor_label] * bursty_len,
                            labels,
                        ]
                    )
                )
            else:
                if target_allowed_in_example:
                    labels = self.rng.choice(
                        self.num_abstract_classes, size=(num_examples)
                    )
                else:
                    # Ensure target is not in example
                    done = False
                    while not done:
                        labels = self.rng.choice(
                            self.num_abstract_classes,
                            size=(num_examples),
                            replace=False,
                        )
                        done = target not in labels

            # Append target to the end
            labels = np.concatenate((labels, [target])).astype(int)
            inputs = np.vstack(list(map(get_input, labels)))

            labels = np.eye(self.num_abstract_classes)[labels]
            yield {
                "example": inputs,
                "label": labels,
            }


def get_dataset(
    num_examples: int,
    p_bursty: int,
    bursty_len: int,
    zipf_exp: float,
    input_noise_std: float,
    target_allowed_in_example: bool = False,
    empty_examples: bool = False,
    num_base_classes: int = 100000,
    num_abstract_classes: int = 32,
    num_dims: int = 128,
    seed: int = 0,
    base_per_abstract_map: str = None,
    novel_abstract_class: bool = False,
):
    task = Reddy(
        num_base_classes,
        num_abstract_classes,
        num_dims,
        seed,
        base_per_abstract_map,
        novel_abstract_class,
    )
    dataset = tf.data.Dataset.from_generator(
        task.get_sequences,
        args=(
            num_examples,
            p_bursty,
            bursty_len,
            zipf_exp,
            input_noise_std,
            target_allowed_in_example,
            empty_examples,
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
