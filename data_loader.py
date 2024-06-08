import os
import tensorflow as tf
import numpy as np
from configuration import *
from utils import visualize
import albumentations as A

DATASET_DIR = os.path.join(os.getcwd(), "data")

SUPPORTED_AUGMENTATIONS = {
    'flip': A.VerticalFlip(p=0.5),
    'horizontal_flip': A.HorizontalFlip(p=0.5),
    'vertical_flip': A.VerticalFlip(p=0.2)
}


class FireDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, batch_size, subset="train", shuffle=True, augmentation=None):
        self.dataset_dir = DATASET_DIR
        self.batch_size = 1 if subset.lower() == "test" else batch_size

        assert subset.lower() in ['train', 'valid', 'test']

        self.subset = subset
        self.shuffle = shuffle

        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH

        self.augmentation = augmentation

        dataset_path = os.path.join(self.dataset_dir, f"{subset.lower()}/")

        image_paths = []
        gt_paths = []
        indexes = []

        file_names = tf.data.Dataset.list_files(dataset_path + "*_mask.png", shuffle=False)

        img_num = 0
        for file_name in file_names:
            indexes.append(img_num)

            image_name = file_name.numpy().decode("utf-8")[len(dataset_path):-9]

            img_path = dataset_path + image_name + ".jpg"
            ann_path = dataset_path + image_name + "_mask.png"

            image_paths.append(img_path)
            gt_paths.append(ann_path)

            img_num += 1

        self.indexes = np.array(indexes)
        self.data_len = len(self.indexes)
        self.image_paths = image_paths
        self.gt_paths = gt_paths

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        def add_sample_weights(gt):
            class_weights = tf.constant([1.0, 2.0])
            class_weights = class_weights / tf.reduce_sum(class_weights)
            sample_weights = tf.where(tf.equal(gt, 0), class_weights[0], class_weights[1])
            return sample_weights

        def decode_img(image, channels):
            img = tf.io.decode_png(image, channels=channels, dtype=tf.uint8)
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.resize(img, [self.img_height, self.img_width])
            if channels == 1:
                img = tf.math.ceil(img)
            #                img = tf.cast(img, dtype=tf.uint8)
            return img

        def process_path(img_path, gt_path):
            img = tf.io.read_file(img_path)
            gt = tf.io.read_file(gt_path)
            img = decode_img(img, channels=3)
            gt = decode_img(gt, channels=1)
            if self.augmentation:
                sample = self.augmentation(image=img.numpy(), mask=gt.numpy())
                img = sample['image']
                gt = sample['mask']
            return tf.concat([img, gt], axis=2)

        selected_image_paths = [self.image_paths[i] for i in batch_indexes]
        selected_gt_paths = [self.gt_paths[i] for i in batch_indexes]

        image_gt_batch = tf.stack([process_path(img_path, gt_path) for
                                   img_path, gt_path in zip(selected_image_paths, selected_gt_paths)])

        images = image_gt_batch[:, :, :, 0: 3]
        gts = tf.cast(image_gt_batch[:, :, :, -1], tf.uint8)

        return images, gts

    def __len__(self):
        return self.data_len // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(self.data_len)
        if self.shuffle:
            np.random.permutation(self.indexes)

    def full_length(self):
        return self.data_len


def get_fire_data_generator(batch_size, subset, shuffle, augmentation_dict=None):
    augmentations = []
    if augmentation_dict:
        if subset.lower() in ["train"]:
            assert all(key in SUPPORTED_AUGMENTATIONS.keys() for key in augmentation_dict.keys()), \
                (f"The following augmentations are not supported: "
                 f"{', '.join(set(augmentation_dict.keys()) - set(SUPPORTED_AUGMENTATIONS.keys()))}")

            for aug_name, aug_value in augmentation_dict.items():
                assert 0 <= aug_value <= 1
                aug = SUPPORTED_AUGMENTATIONS[aug_name]
                aug.p = aug_value
                augmentations.append(SUPPORTED_AUGMENTATIONS[aug_name])

            if len(augmentations) > 0:
                augmentations = A.Compose(augmentations)
            else:
                augmentations = None
        else:
            assert False, "Data augmentation is only supported for training data."

    fire_generator = FireDataGenerator(batch_size=batch_size,
                                       subset=subset,
                                       shuffle=shuffle,
                                       augmentation=augmentations)
    return fire_generator


if __name__ == "__main__":
    example_datagen = FireDataGenerator(subset="train", shuffle=False)
    images, masks = next(iter(example_datagen))
    image_dict = {
        'image': images[0],
        'mask': masks[0]
    }
    visualize(**image_dict)