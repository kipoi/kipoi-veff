Kipoi offers a set of postprocessing tools that enable to calculate variant effects, create mutation maps, inspect
activation of hidden model layers and to calculate the gradient of layer activation with respect to a given input.

Variant effect prediction and mutation map generation is available for all models where the `variant_effects` parameter
 in the model.yaml (and dataloader.yaml) is set (see here)[http://kipoi.org/docs/postprocessing/variant_effect_prediction].
Inspection of the activation of hidden model layers and calculation of gradients is available for all deep learning
models: Currently supported are Keras, PyTorch and Tensorflow models. For a detailed description and examples of how
to use tose features please take a look at:

* [Variant effect prediction](./variant_effect_pred.md)
* [Mutation maps](./mutation_map.md)
