import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from configuration import *
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import yaml
from data_loader import get_fire_data_generator
from callbacks import get_callbacks


def train(input_yaml):
    with open(input_yaml, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    loss_name = config['loss']
    assert loss_name in SUPPORTED_LOSSES.keys(), "Loss not supported or not found."

    metric_name = config['metric']
    assert metric_name in SUPPORTED_METRICS.keys(), "Metric not supported or not found."

    backbone = config['backbone']
    assert backbone in SUPPORTED_BACKBONES, "Backbone not supported or not found."

    optimizer_dict = config['optimizer']
    optimizer_name = optimizer_dict['name']
    assert optimizer_name in SUPPORTED_OPTIMIZERS.keys(), "Optimizer not supported or not found."
    init_lr = optimizer_dict.get('init_lr', 0.001)

    loss = SUPPORTED_LOSSES[loss_name]
    metric = SUPPORTED_METRICS[metric_name]
    optimizer = SUPPORTED_OPTIMIZERS[optimizer_name]
    optimizer.learning_rate = init_lr

    model = sm.Unet(backbone, encoder_weights="imagenet")
    model.compile(optimizer=optimizer, loss=[loss], metrics=[metric])

    # Default basename for output files
    testcase = f"{backbone}_{loss_name}_{metric_name}"

    callback_dict = config.get('callbacks', {})
    callbacks = get_callbacks(callback_dict=callback_dict, output_file_basename=testcase)

    augmentation_dict = config.get('augmentations', {})

    batch_size = config.get("batch_size", 16)
    print("Loading training data . . .")
    train_data = get_fire_data_generator(batch_size=batch_size, subset="train", shuffle=True,
                                         augmentation_dict=augmentation_dict)
    val_data = get_fire_data_generator(batch_size=batch_size, subset="valid", shuffle=False)
    print("Finished loading training data . . .")

    epochs = config.get("epochs", 40)
    steps_per_epoch = config.get("steps_per_epoch", len(train_data))
    validation_steps = config.get("validation_steps", len(val_data))

    print("Begin training . . .")
    history = model.fit(train_data,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        validation_data=val_data,
                        callbacks=callbacks)
    print("Training ended . . .")


if __name__ == "__main__":
    input_yaml_file = "train_configs/train_config_1.yaml"
    train(input_yaml_file)
