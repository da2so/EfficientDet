import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD

from generators.create_generators import create_generators
from core.efficientdet import EfficientDet, EfficientDet_P
from core.loss import smooth_l1, focal, smooth_l1_quad
from core.utils import get_callbacks

class Trainer():
    def __init__(self, 
                coco_path, 
                dataset_type, 
                random_transform,
                bs, 
                steps,
                epochs,
                phi, 
                num_workers,
                use_multiprocessing, 
                max_queue_size,
                compute_val_loss,
                save_path
                ):
        # for multi-gpu training
        self.strategy = tf.distribute.MirroredStrategy()
        self.num_devices = int(self.strategy.num_replicas_in_sync)

        self.dataset_type =dataset_type
        self.bs = bs * self.num_devices
        self.compute_val_loss = compute_val_loss
        self.steps = int(steps / self.num_devices)
        self.epochs = epochs
        self.phi = phi
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size
        self.save_path = save_path


        print ('Number of devices: {}'.format(self.num_devices))

        self.train_generator, self.validation_generator = create_generators(coco_path = coco_path,
                                                                            dataset_type = self.dataset_type, 
                                                                            bs = self.bs, 
                                                                            phi = self.phi,
                                                                            random_transform =  random_transform)
        
        self.num_classes = self.train_generator.num_classes()
        self.num_anchors = self.train_generator.num_anchors
        with self.strategy.scope():
            self.efficientdet_train_model = EfficientDet(phi = self.phi, 
                                                         num_classes = self.num_classes,
                                                         num_anchors = self.num_anchors,
                                                         strategy = self.strategy)
            self.efficientdet_train_model.compile(optimizer = Adam(lr=1e-3), 
                                                  loss = {'regression': smooth_l1(),
                                                          'classification': focal()},)

        self.efficientdet_predict_model = EfficientDet_P(phi = self.phi, 
                                                         efficientdet_model = self.efficientdet_train_model)


        self.callbacks = get_callbacks(model = self.efficientdet_predict_model,
                                       validation_generator = self.validation_generator,
                                       dataset_type = self.dataset_type,
                                       compute_val_loss = self.compute_val_loss,
                                       save_path = self.save_path,)
    def run(self):

        self.efficientdet_train_model.fit(x = self.train_generator,
                                          batch_size = self.bs,
                                          epochs = self.epochs,
                                          verbose = 1,
                                          callbacks = self.callbacks,
                                          steps_per_epoch = self.steps,
                                          workers = self.num_workers,
                                          use_multiprocessing= self.use_multiprocessing,
                                          max_queue_size=self.max_queue_size)