import keras
import pytest

from . import analyze as module


class TestGetSubmodelForInput:
    def test_extracts_submodel_for_sequential_model(self):
        model = keras.Sequential(
            [
                keras.layers.Dense(1, input_shape=(1,), name="first_dense"),
                keras.layers.Dense(1, name="stop_here"),
                keras.layers.Dense(1, name="layer_to_exclude"),
            ]
        )

        submodel = module.get_submodel_for_input(
            model, input_index=0, last_layer="stop_here"
        )

        expected_layers = set(["first_dense_input", "first_dense", "stop_here"])

        assert expected_layers == set([layer.name for layer in submodel.layers])

    @pytest.mark.parametrize("input_index", (0, 1))
    def test_extracts_submodel_for_multi_branch_model(self, input_index):
        first_branch_input = keras.layers.Input(shape=(1,), name="input_0")
        first_branch_dense = keras.layers.Dense(1, name="dense_0_1")(first_branch_input)

        second_branch_input = keras.layers.Input(shape=(1,), name="input_1")
        second_branch_dense = keras.layers.Dense(1, name="dense_1_1")(
            second_branch_input
        )

        concated_branches = keras.layers.concatenate(
            [first_branch_dense, second_branch_dense]
        )

        model = keras.models.Model(
            inputs=[first_branch_input, second_branch_input],
            outputs=[concated_branches],
        )

        last_layer = f"dense_{input_index}_1"

        submodel = module.get_submodel_for_input(
            model, input_index=input_index, last_layer=last_layer
        )

        expected_layers = set([f"input_{input_index}", last_layer])

        assert expected_layers == set([layer.name for layer in submodel.layers])
