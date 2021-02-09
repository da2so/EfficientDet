import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from core.train import Trainer


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='EfficientDet Tensorflow Keras')

    # Arguments related to dataset
    parser.add_argument('--coco_path', type = str, default = './dataset/coco/', help ='coco dataset path' )
    parser.add_argument('--dataset_type', type = str, default = 'coco', help ='dataset type' )
    parser.add_argument('--random_transform', type = str2bool, default = 'True', help = 'Randomly transform image and annotations')

    # Arguments related to train config
    parser.add_argument('--bs', type = int, default = 8, help = 'Batch size')
    parser.add_argument('--steps', type=int, default = 10000, help = 'Number of steps per epoch.')
    parser.add_argument('--epochs', type = int, default = 50, help = 'Epoch number')

    # Argument related to model
    parser.add_argument('--phi', type = int, default = 0, choices=(0, 1, 2, 3, 4, 5, 6), help = 'Hyper parameter phi')

    # Arguments related to generator config
    parser.add_argument('--num_workers', type = int, default = 1, help = 'Number of generator workers')
    parser.add_argument('--use_multiprocessing', type = str2bool, default = 'True', help = 'Use multiprocessing?')
    parser.add_argument('--max_queue_size', type = int, default = 10, help = 'Queue length for multiprocessing workers')

    # Arguments related to callback list
    parser.add_argument('--compute_val_loss', type = str2bool, default = 'True', help = 'Compute validation loss?')
    parser.add_argument('--save_path', type = str, default = './result/', help = 'Save path for the object detection model')


    args = parser.parse_args()

    trainer_obj = Trainer(  coco_path           = args.coco_path,
                            dataset_type        = args.dataset_type,
                            random_transform    = args.random_transform,
                            bs                  = args.bs,
                            steps               = args.steps,
                            epochs              = args.epochs,
                            phi                 = args.phi,
                            num_workers         = args.num_workers,
                            use_multiprocessing = args.use_multiprocessing,
                            max_queue_size      = args.max_queue_size,
                            compute_val_loss    = args.compute_val_loss,
                            save_path           = args.save_path )
    trainer_obj.run()

