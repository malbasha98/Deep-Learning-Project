import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from utils import compute_sp_iou, extract_bboxes, visualize_images_and_masks
from configuration import *
import segmentation_models as sm
from data_loader import get_fire_data_generator

sm.set_framework('tf.keras')


def evaluate_image(filepath, pred_mask):
    basename = os.path.basename(filepath)[:-4]
    fire_sp_paths = glob.glob(TEST_FIRE_SP_DIR + f"/{basename}sp*")
    nofire_sp_paths = glob.glob(TEST_NOFIRE_SP_DIR + f"/{basename}sp*")

    fire_classification_results = []
    fire_true = np.ones(len(fire_sp_paths))
    nofire_classification_results = []
    nofire_true = np.ones(len(nofire_sp_paths))

    if nofire_sp_paths:
        for fire_sp_path in fire_sp_paths:
            nofire_sp = cv2.imread(fire_sp_path, cv2.IMREAD_GRAYSCALE)
            nofire_sp = cv2.resize(nofire_sp, (IMG_WIDTH, IMG_HEIGHT))
            bb_coordinates = extract_bboxes(nofire_sp)
            y1, y2, x1, x2 = bb_coordinates[0], bb_coordinates[2], bb_coordinates[1], bb_coordinates[3]
            nofire_sp = nofire_sp[y1: y2, x1: x2]
            pred_sp = 255 - nofire_sp[y1: y2, x1: x2]
            iou = compute_sp_iou(nofire_sp, pred_sp)
            if iou > 0.5:
                nofire_classification_results.append(True)
            else:
                nofire_classification_results.append(False)
    if fire_sp_paths:
        for fire_sp_path in fire_sp_paths:
            fire_sp = cv2.imread(fire_sp_path, cv2.IMREAD_GRAYSCALE)
            fire_sp = cv2.resize(fire_sp, (IMG_WIDTH, IMG_HEIGHT))
            bb_coordinates = extract_bboxes(fire_sp)
            y1, y2, x1, x2 = bb_coordinates[0], bb_coordinates[2], bb_coordinates[1], bb_coordinates[3]
            fire_sp = fire_sp[y1: y2, x1: x2]
            pred_sp = fire_sp[y1: y2, x1: x2]
            iou = compute_sp_iou(fire_sp, pred_sp)
            if iou > 0.5:
                fire_classification_results.append(True)
            else:
                fire_classification_results.append(False)

    fire_classification_results = np.array(fire_classification_results)
    nofire_classification_results = np.array(nofire_classification_results)

    tp = np.sum(fire_classification_results * fire_true)
    tn = np.sum(nofire_classification_results * nofire_true)
    fp = np.logical_xor(fire_classification_results, fire_true)
    fn = np.logical_xor(nofire_classification_results, nofire_true)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return np.array([tpr, fpr, precision, recall, accuracy], dtype="float32")


if __name__ == "__main__":
    backbone_name = "resnet101"
    h5_weights = os.path.join(os.getcwd(), "outputs", "UNET weights", "resnet101_bce_dice_iou.h5")
    eval_model = sm.Unet(backbone_name)
    eval_model.load_weights(h5_weights)


