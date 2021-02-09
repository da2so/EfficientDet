"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# import keras
from tensorflow import keras
import tensorflow as tf

from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from tqdm import trange
import cv2
import os

from generators.coco import CocoGenerator


def evaluate(generator, model, save_path, threshold=0.01):
    """
    Use the pycocotools to evaluate a COCO model on a dataset.
    Args
        generator: The generator for generating the evaluation data.
        model: The model to evaluate.
        threshold: The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    for index in trange(generator.size(), desc='COCO evaluation: '):
        image = generator.load_image(index)

        src_image = image.copy()
        h, w = image.shape[:2]
        image, scale = generator.preprocess_image(image)

        # run network
        _, _, detections = model.predict_on_batch([np.expand_dims(image, axis=0)])


        boxes, scores, labels = detections
        boxes /= scale
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > threshold)[0]
        boxes = boxes[0, indices]
        scores = scores[0, indices]
        class_ids = labels[0, indices]

        # compute predicted labels and scores
        for box, score, class_id in zip(boxes, scores, class_ids):
            # append detection for each positively labeled class
            image_result = {
                'image_id': generator.image_ids[index],
                'category_id': int(class_id) + 1,
                'score': float(score),
                'bbox': box.tolist(),
            }
            # append detection to results
            results.append(image_result)
        #     box = np.round(box).astype(np.int32)
        #     class_name = generator.label_to_name(generator.coco_label_to_label(class_id + 1))
        #     ret, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     cv2.rectangle(src_image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
        #     cv2.putText(src_image, class_name, (box[0], box[1] + box[3] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (0, 0, 0), 1)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', src_image)
        # cv2.waitKey(0)

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])

    if not len(results):
        return

    # write output
    json.dump(results, open('{}{}_bbox_results.json'.format(save_path, generator.set_name), 'w'), indent=4)
    json.dump(image_ids, open('{}{}_processed_image_ids.json'.format(save_path, generator.set_name), 'w'), indent=4)

    # # load results in COCO evaluation tool
    coco_true = generator.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(generator.set_name))
    
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


class Evaluate(keras.callbacks.Callback):
    """ Performs COCO evaluation on each epoch.
    """

    def __init__(self, generator, model, save_path, threshold=0.01):
        super(Evaluate, self).__init__()
        """ Evaluate callback initializer.
        Args
            generator : The generator used for creating validation data.
            model: prediction model
            threshold : The score threshold to use.
        """
        self.generator = generator
        self.active_model = model
        self.threshold = threshold
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        coco_tag = ['AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.75      | area=   all | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                    'AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                    'AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']

        file_dir = f'{self.save_path}{epoch}/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_path = f'{file_dir}result.txt'
        f = open(file_path, 'w')

        result_stat = evaluate(self.generator, self.active_model,  file_dir, self.threshold)


        for index, result in enumerate(result_stat):
            tag = '{}. {} = {}'.format(index + 1, coco_tag[index], result) 
            f.write(tag)
        
        f.close()  
