from evaluate_summarized import *
from configuration import OUTPUT_DIR
import glob


def evaluate_all():
    weights_dir = os.path.join(OUTPUT_DIR, "UNET weights")
    if not os.path.exists(weights_dir):
        print("No availale UNET weights!")
        return
    else:
        h5_files = [f for f in os.listdir(weights_dir) if f.endswith(".h5")]
        if not h5_files:
            print(f"The directory '{weights_dir}' does not contain .h5 files:")
            return

    h5_filepaths = glob.glob(weights_dir + "/*.h5")
    print("Number of UNET weights found: ", len(h5_filepaths))
    for h5_filepath in h5_filepaths:
        evaluate_on_all_images(h5_filepath)


if __name__ == "__main__":
    evaluate_all()

