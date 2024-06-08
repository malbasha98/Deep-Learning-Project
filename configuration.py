import os
import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models.metrics import  *
from segmentation_models.losses import  *
import albumentations as A

IMG_HEIGHT = 224
IMG_WIDTH = 224

TEST_FIRE_SP_DIR = os.path.join(os.getcwd(), "data/eval/superpixels/isolated-superpixels/test/fire")
TEST_NOFIRE_SP_DIR = os.path.join(os.getcwd(), "data/eval/superpixels/isolated-superpixels/test/nofire")
TEST_IMAGES_DIR = os.path.join(os.getcwd(), "data/eval/superpixels/original-full-images/test")

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' created successfully.")
    except OSError as e:
        print(f"Error occurred while creating directory '{OUTPUT_DIR}': {e}")

SUPPORTED_LOSSES = {
    'jaccard': JaccardLoss(),
    'dice': DiceLoss(),
    'bce': BinaryCELoss(),
    'bce_dice': BinaryCELoss() + DiceLoss(),
    'bce_jaccard': BinaryCELoss() + JaccardLoss()
}

SUPPORTED_METRICS = {
    'iou': IOUScore(),
    'precission': Precision(),
    'recall': Recall()
}

SUPPORTED_OPTIMIZERS = {
    'adam': tf.keras.optimizers.Adam()
}

SUPPORTED_BACKBONES = ['vgg19', 'resnet101', 'inceptionv3', 'mobilenetv2', 'efficientnetb5']

SUPPORTED_AUGMENTATIONS = {
    'flip': A.VerticalFlip(p=0.5),
    'horizontal_flip': A.HorizontalFlip(p=0.5),
    'vertical_flip': A.VerticalFlip(p=0.2)
}