import os
import tensorflow as tf


def get_callbacks(callback_dict, output_file_basename):
    callbacks = []

    csv_logger_dict = callback_dict.get("csv_logger", {})
    if csv_logger_dict:
        assert "output_csv_file" in csv_logger_dict.keys(), ("Enter the path to save training results for csv "
                                                             "logger callback!")
        output_csv_file = output_file_basename + ".csv" if csv_logger_dict['output_csv_file'] == "default" \
            else csv_logger_dict['output_csv_file']
        output_csv_dir = os.path.join(os.getcwd(), "outputs", "training results")
        output_csv_path = os.path.join(output_csv_dir, output_csv_file)
        csv_logger = tf.keras.callbacks.CSVLogger(output_csv_path)
        callbacks.append(csv_logger)

    early_stopping_dict = callback_dict.get('early_stopping', {})
    if early_stopping_dict:
        assert "patience" in early_stopping_dict.keys(), "Enter patience for early stopping callback!"
        assert "monitor" in early_stopping_dict.keys(), "Enter value to monitor for early stopping callback!"
        assert "mode" in early_stopping_dict.keys(), "Enter mode (min or max) for early stopping callback!"

        patience = early_stopping_dict['patience']
        monitor = early_stopping_dict['monitor']
        mode = early_stopping_dict['mode']
        assert mode in ['min', 'max']

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, verbose=1,
                                                          patience=patience, restore_best_weights=True)
        callbacks.append(early_stopping)

    reduce_lr_dict = callback_dict.get('reduce_lr_on_plateau', {})
    if reduce_lr_dict:
        assert "patience" in reduce_lr_dict.keys(), "Enter patience for reduce lr on plateau callback!"
        assert "monitor" in reduce_lr_dict.keys(), "Enter value to monitor for reduce lr on plateau callback!"
        assert "mode" in reduce_lr_dict.keys(), "Enter mode (min or max) for reduce lr on plateau callback!"

        patience = reduce_lr_dict['patience']
        monitor = reduce_lr_dict['monitor']
        mode = reduce_lr_dict['mode']
        factor = reduce_lr_dict['factor']
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode, verbose=1,
                                                         factor=factor, patience=patience)
        callbacks.append(reduce_lr)

    save_model_dict = callback_dict.get('save_model', {})
    if save_model_dict:
        assert "monitor" in save_model_dict.keys(), "Enter value to monitor for model checkpoint callback!"
        assert "mode" in save_model_dict.keys(), "Enter mode (min or max) for model checkpoint callback!"
        assert "mode" in save_model_dict.keys(), "Enter path to save weights for model checkpoint callback!"

        monitor = save_model_dict['monitor']
        mode = save_model_dict['mode']
        unet_weights_file = output_file_basename + ".h5" if save_model_dict['unet_weights_file'] == "default" \
            else save_model_dict['unet_weights_file']
        output_unet_weights_dir = os.path.join(os.getcwd(), "outputs", "UNET weights")
        unet_weights_path = os.path.join(output_unet_weights_dir, unet_weights_file)
        save_model = tf.keras.callbacks.ModelCheckpoint(unet_weights_path, monitor=monitor,
                                                        mode=mode, verbose=1,
                                                        save_best_only=True)
        callbacks.append(save_model)

    print("Using callbacks: ", [callback_name.title() for callback_name in callback_dict.keys()
                                if callback_name is not None])

    return callbacks
