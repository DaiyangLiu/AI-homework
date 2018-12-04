import os

from _2_train_isensee2017 import config
from unet3d.prediction import run_validation_cases

from mgpu_for_train import set_gpus
def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir)


if __name__ == "__main__":
    set_gpus(num_gpus=1, free_ratio=1.0, auto_growth=False)
    main()
