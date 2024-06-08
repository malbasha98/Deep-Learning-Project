import matplotlib.pyplot as plt
import numpy as np


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def visualize_images_and_masks(images, masks, rows=3, cols=4, figsize=(12, 8)):
    num_images = len(images)
    if len(masks) != num_images:
        raise ValueError("The number of images and masks must be the same.")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, (image, mask) in enumerate(zip(images, masks)):
        if i >= rows * cols:
            break

        axes[i].imshow(image)
        axes[i].imshow(mask, alpha=0.5)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def extract_bboxes(mask):
    box = np.zeros([1, 4], dtype=np.int32)
    m = mask[:, :]
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        x2 += 1
        y2 += 1
    else:
        x1, x2, y1, y2 = 0, 0, 0, 0
    box = np.array([y1, x1, y2, x2])
    return box.astype(np.int32)


def compute_sp_iou(true_sp, pred_sp):
    true_sp = true_sp.flatten() / 255
    pred_sp = pred_sp.flatten() / 255

    intersection = len(np.where(true_sp * pred_sp > 0)[0])
    union = np.sum(np.logical_or(true_sp, pred_sp))
    print(union)

    return intersection / union
