import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import segmentation_models as sm
# sm.set_framework('tf.keras')
from data_loader import get_fire_data_generator
from configuration import *
from utils import compute_sp_iou, extract_bboxes, visualize_images_and_masks

captured_image = []


def evaluate_on_all_images(h5_weights_path, fire_iou_thres=0.35, nofire_iou_thres=0.5):
    df_data = []
    basename = os.path.basename(h5_weights_path)
    basename_split = basename.split("_")
    backbone = basename_split[0]
    loss = basename_split[1] if len(basename_split) == 3 else basename_split[1] + "_" + basename_split[2]
    metric = basename_split[2] if len(basename_split) == 3 else basename_split[1] + "_" + basename_split[3]

    eval_output_dir = os.path.join(OUTPUT_DIR, "evaluation results")
    if not os.path.exists(eval_output_dir):
        try:
            os.makedirs(eval_output_dir)
            print(f"Directory '{eval_output_dir}' created successfully.")
        except OSError as e:
            print(f"Error occurred while creating directory '{eval_output_dir}': {e}")

    eval_model = sm.Unet(backbone)
    eval_model.load_weights(h5_weights_path)

    filepaths = glob.glob(TEST_IMAGES_DIR + "/*.png")
    print(f"Number of filepaths is {len(filepaths)}")

    thresholds = [0.05, 0.2, 0.4]
    for threshold in thresholds:
        print(f"Thresholding UNET outputs with value: {threshold}")
        all_results = []
        for filepath in filepaths:
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
            image = np.array(image, dtype="float32")

            pred = eval_model.predict(np.expand_dims(image, axis=0)).squeeze()
            pred = np.where(pred >= 0.5, 1, 0).astype("uint8")

            results = evaluate_image(filepath, pred, fire_iou_thres, nofire_iou_thres)

            all_results.append(results)

        all_results = np.stack(all_results, axis=0)
        sum_results = np.sum(all_results, axis=0)
        tp, tn, fp, fn = sum_results

        tpr = 0 if tp == 0 else tp / (tp + fn)
        fpr = 0 if fp == 0 else fp / (fp + tn)
        precision = 0 if tp == 0 else tp / (tp + fp)
        recall = 0 if tp == 0 else tp / (tp + fn)
        accuracy = 0 if tp + tn + fp + fn == 0 else (tp + tn) / (tp + tn + fp + fn)

        df_data.append([tp, tn, fp, fn, tpr, fpr, precision, recall, accuracy, threshold])

    df = pd.DataFrame(data= df_data,
                      columns=['tp', 'tn', 'fp', 'fn', 'tpr', 'fpr', 'precision', 'recall', 'accuracy', 'unet_prediction_threshold'])
    df.to_csv(eval_output_dir + f"/{backbone}_{loss}_{metric}_{fire_iou_thres}_{nofire_iou_thres}_summarized_best_2.csv", index=False)


def evaluate_image(filepath, pred_mask, fire_iou_thres=0.35, no_fire_iou_thres=0.5):
    basename = os.path.basename(filepath)[:-4]

    fire_sp_paths = glob.glob(TEST_FIRE_SP_DIR + f"/{basename}sp*")
    nofire_sp_paths = glob.glob(TEST_NOFIRE_SP_DIR + f"/{basename}sp*")

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    if nofire_sp_paths:
        for nofire_sp_path in nofire_sp_paths:
            nofire_sp = cv2.imread(nofire_sp_path, cv2.IMREAD_GRAYSCALE)
            nofire_sp = cv2.resize(nofire_sp, (IMG_WIDTH, IMG_HEIGHT))
            nofire_sp = (nofire_sp > 0).astype(np.uint8)

            pred_sp = (pred_mask > 0).astype(np.uint8)
            pred_sp = pred_sp & nofire_sp

            if np.sum(nofire_sp) == 0:
                continue

            iou_inv = np.sum(pred_sp) / np.sum(nofire_sp)

            if iou_inv <= no_fire_iou_thres:
                tn += 1
            else:
                fp += 1

    if fire_sp_paths:
        for fire_sp_path in fire_sp_paths:
            fire_sp = cv2.imread(fire_sp_path, cv2.IMREAD_GRAYSCALE)
            fire_sp = cv2.resize(fire_sp, (IMG_WIDTH, IMG_HEIGHT))
            fire_sp = (fire_sp > 0).astype(np.uint8)

            pred_sp = (pred_mask > 0).astype(np.uint8)
            pred_sp = pred_sp & fire_sp

            iou = np.sum(pred_sp) / np.sum(fire_sp)

            if iou >= fire_iou_thres:
                tp += 1
            else:
                fn += 1

    return np.array([tp, tn, fp, fn])


if __name__ == "__main__":
    h5_weights = os.path.join(os.getcwd(), "outputs", "UNET weights", "resnet101_bce_jaccard_iou.h5")
    evaluate_on_all_images(h5_weights)
    # h5_weights = os.path.join(os.getcwd(), "outputs", "UNET weights", "vgg19_dice_iou.h5")
    # evaluate_on_all_images(h5_weights)

