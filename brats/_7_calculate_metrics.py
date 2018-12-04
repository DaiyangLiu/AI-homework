import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Dice_WT Dice_CT Dice_ET Sensitivity_WT Sensitivity_CT Sensitivity_ET Specificity_WT Specificity_CT Specificity_ET Hausdorff_WT Hausdorff_CT Hausdorff_ET
    brats_evaluate_csv = pd.read_csv("./prediction/brats_evaluate.csv")

    evaluate_brats_2017_txt = open("./prediction/eval_brats_2017.txt","w+")
    evaluate_brats_2017_txt.write("Evaluate metrics:\n")
    evaluate_brats_2017_txt.write("Dice_WT mean: "+ str(np.mean(np.array(brats_evaluate_csv["Dice_WT"])))+", Dice_WT std: "+ str(np.std(np.array(brats_evaluate_csv["Dice_WT"]))) +"\n")
    evaluate_brats_2017_txt.write("Dice_CT mean: " + str(np.mean(np.array(brats_evaluate_csv["Dice_CT"]))) + ", Dice_CT std: " + str(np.std(np.array(brats_evaluate_csv["Dice_CT"]))) + "\n")
    evaluate_brats_2017_txt.write("Dice_ET mean: " + str(np.mean(np.array(brats_evaluate_csv["Dice_ET"]))) + ", Dice_ET std: " + str(np.std(np.array(brats_evaluate_csv["Dice_ET"]))) + "\n")
    evaluate_brats_2017_txt.write("Sensitivity_WT mean: " + str(np.mean(np.array(brats_evaluate_csv["Sensitivity_WT"]))) + ", Sensitivity_WT std: " + str(np.std(np.array(brats_evaluate_csv["Sensitivity_WT"]))) + "\n")
    evaluate_brats_2017_txt.write("Sensitivity_CT mean: " + str(np.mean(np.array(brats_evaluate_csv["Sensitivity_CT"]))) + ", Sensitivity_CT std: " + str(np.std(np.array(brats_evaluate_csv["Sensitivity_CT"]))) + "\n")
    evaluate_brats_2017_txt.write("Sensitivity_ET mean: " + str(np.mean(np.array(brats_evaluate_csv["Sensitivity_ET"]))) + ", Sensitivity_ET std: " + str(np.std(np.array(brats_evaluate_csv["Sensitivity_ET"]))) + "\n")
    evaluate_brats_2017_txt.write("Specificity_WT mean: " + str(np.mean(np.array(brats_evaluate_csv["Specificity_WT"]))) + ", Specificity_WT std: " + str(np.std(np.array(brats_evaluate_csv["Specificity_WT"]))) + "\n")
    evaluate_brats_2017_txt.write("Specificity_CT mean: " + str(np.mean(np.array(brats_evaluate_csv["Specificity_CT"]))) + ", Specificity_CT std: " + str(np.std(np.array(brats_evaluate_csv["Specificity_CT"]))) + "\n")
    evaluate_brats_2017_txt.write("Specificity_ET mean: " + str(np.mean(np.array(brats_evaluate_csv["Specificity_ET"]))) + ", Specificity_ET std: " + str(np.std(np.array(brats_evaluate_csv["Specificity_ET"]))) + "\n")
    evaluate_brats_2017_txt.write("Hausdorff_WT mean: " + str(np.mean(np.array(brats_evaluate_csv["Hausdorff_WT"]))) + ", Hausdorff_WT std: " + str(np.std(np.array(brats_evaluate_csv["Hausdorff_WT"]))) + "\n")
    evaluate_brats_2017_txt.write("Hausdorff_CT mean: " + str(np.mean(np.array(brats_evaluate_csv["Hausdorff_CT"]))) + ", Hausdorff_CT std: " + str(np.std(np.array(brats_evaluate_csv["Hausdorff_CT"]))) + "\n")
    evaluate_brats_2017_txt.write("Hausdorff_ET mean: " + str(np.mean(np.array(brats_evaluate_csv["Hausdorff_ET"]))) + ", Hausdorff_ET std: " + str(np.std(np.array(brats_evaluate_csv["Hausdorff_ET"]))) + "\n")

    evaluate_brats_2017_txt.close()