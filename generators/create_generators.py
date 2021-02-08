
from augmentors.color import VisualEffect
from augmentors.misc import MiscEffect
from generators.coco import CocoGenerator

def create_generators(coco_path, dataset_type, bs, phi, random_transform):
    """
    Create generators for training and validation.
    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': bs,
        'phi': phi,
        }

    # create random transform generator for augmenting training data
    if random_transform:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None


    if dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        train_generator = CocoGenerator(
            coco_path,
            'train2017',
            misc_effect = misc_effect,
            visual_effect = visual_effect,
            group_method='random',
            **common_args
        )

        validation_generator = CocoGenerator(
            coco_path,
            'val2017',
            shuffle_groups = False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator