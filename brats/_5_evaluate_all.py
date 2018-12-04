import numpy as np
import random
import glob
import os
import SimpleITK as sitk
from evaluation_metrics import *
import pandas as pd
import nibabel as nib

# def get_whole_tumor_mask(data):
#     return data > 0
#
#
# def get_tumor_core_mask(data):
#     return np.logical_or(data == 1, data == 4)
#
#
# def get_enhancing_tumor_mask(data):
#     return data == 4
#
#
# def dice_coefficient(truth, prediction):
#     return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


def main():
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    #masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()
    for case_folder in glob.glob("prediction/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()

        # compute the evaluation metrics
        Dice_complete = DSC_whole(prediction, truth)
        Dice_enhancing = DSC_en(prediction, truth)
        Dice_core = DSC_core(prediction, truth)

        Sensitivity_whole = sensitivity_whole(prediction, truth)
        Sensitivity_en = sensitivity_en(prediction, truth)
        Sensitivity_core = sensitivity_core(prediction, truth)

        Specificity_whole = specificity_whole(prediction, truth)
        Specificity_en = specificity_en(prediction, truth)
        Specificity_core = specificity_core(prediction, truth)

        Hausdorff_whole = hausdorff_whole(prediction, truth)
        Hausdorff_en = hausdorff_en(prediction, truth)
        Hausdorff_core = hausdorff_core(prediction, truth)

        tmp = np.array((Dice_complete,Dice_core,Dice_enhancing,Sensitivity_whole,Sensitivity_core,Sensitivity_en,Specificity_whole,Specificity_core,Specificity_en,Hausdorff_whole,Hausdorff_core,Hausdorff_en))

        rows.append(tmp)

    eval_results_data = np.array(rows)
        # # Dice_WT Dice_CT Dice_ET  Sensitivity_WT Sensitivity_CT Sensitivity_ET Specificity_WT Specificity_CT Specificity_ET Hausdorff_WT Hausdorff_CT Hausdorff_ET

    eval_results_csv = pd.DataFrame(columns=["Dice_WT", "Dice_CT", "Dice_ET",
                                                 "Sensitivity_WT", "Sensitivity_CT", "Sensitivity_ET",
                                                 "Specificity_WT", "Specificity_CT", "Specificity_ET",
                                                 "Hausdorff_WT", "Hausdorff_CT", "Hausdorff_ET"
                                                 ],
                                        data=eval_results_data,index=subject_ids)


    eval_results_csv.to_csv("./prediction/brats_evaluate.csv", index=False)


        #rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    # df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    # df.to_csv("./prediction/brats_scores_all.csv")



if __name__ == "__main__":
    main()