import chex


class TFDataset:
    def __init__(self, dataset, num_classes, input_shape, sequence_length):
        self._dataset = dataset
        self._num_classes = num_classes
        self._input_shape = input_shape
        self._sequence_length = sequence_length

    @property
    def dataset(self):
        return self._dataset

    @property
    def output_dim(self) -> chex.Array:
        return (self._num_classes,)

    @property
    def input_dim(self) -> chex.Array:
        return self._input_shape

    @property
    def sequence_length(self) -> int:
        return self._sequence_length
