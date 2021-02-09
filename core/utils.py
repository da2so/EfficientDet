import numpy as np
import math
import os

from tensorflow import keras

from core import backbone

class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability = 0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype = None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype = np.float32) * -math.log((1 - self.probability) / self.probability)

        return result


def get_backbone(model_name, input_tensor, strategy):
    
    try:
        model_func = getattr(backbone, model_name) 
    except:
        raise ValueError('Invalid model name : {}'.format(model_name))
    
    model = model_func(input_tensor = input_tensor, strategy = strategy)
    return model

def get_callbacks(model,
                validation_generator, 
                dataset_type,
                compute_val_loss,
                save_path = None,
                ):

    callbacks = []

    if compute_val_loss and validation_generator and save_path != None:
        if dataset_type == 'coco':
            from eval.coco import Evaluate
            # use prediction model for evaluation
            evaluation = Evaluate(validation_generator, model, save_path)
        else:
            from eval.pascal import Evaluate
            evaluation = Evaluate(validation_generator, model, save_path)
        callbacks.append(evaluation)

    # save the model
    if save_path != None:
        # ensure directory created first; otherwise h5py will error after epoch.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            f'{save_path}{{epoch:02d}}/{dataset_type}_{{loss:.4f}}.h5',
            verbose=1,
            save_weights_only=True,
        )
        callbacks.append(checkpoint)

    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor='loss',
    #     factor=0.1,
    #     patience=2,
    #     verbose=1,
    #     mode='auto',
    #     min_delta=0.0001,
    #     cooldown=0,
    #     min_lr=0
    # ))

    return callbacks
