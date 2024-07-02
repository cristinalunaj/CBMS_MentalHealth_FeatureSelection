import pandas as pd
import numpy as np
import os
from math import log
import matplotlib.pyplot as plt
from src.models.embs_Experiments.feature_selection.featureSelection import create_plot_dims
pd.set_option('display.max_columns', 6)


# calculate aic for regression
def calculate_aic(n, performance, num_params):
 aic = n * log(performance) + 2 * num_params
 return aic



if __name__ == '__main__':
    # parse input arguments
    DS = "counseilChat" #counseilChat 7cups
    MODEL_PATH = "klyang_MentaLLaMA-chat-7B" # klyang_MentaLLaMA-chat-7B meta-llama_Llama-2-7b-chat-hf
    common_PATH = "../../CBMS_MentalHealth_FeatureSelection/data/experiments/featureSelection/"+MODEL_PATH+"/"+DS+"/ANOVA-F"



    path_features_ANOVA = os.path.join(common_PATH, "statistics_ANOVA-F.csv")
    path_models_FS = os.path.join(common_PATH, "_all_TopicClassification.csv_norm2_out_FS_simpleClassifiers.csv")

    df_ANOVA_ranking = pd.read_csv(path_features_ANOVA, sep=";", header=0)
    df_outputs = pd.read_csv(path_models_FS, sep=";", header=0)

    met = "weighted_f1_" #weighted_f1_ macro_f1_
    raio_name = "ratioWF1_"
    metric_name = met[0:-1]#raio_name[0:-1]

    if(DS=="7cups"):
        N_samples_train,N_samples_dev,N_samples_test = 109076, 15466, 17688
    else:
        N_samples_train, N_samples_dev, N_samples_test = 2545, 419, 487
    N_samples = {"train": N_samples_train,
                 "dev": N_samples_dev,
                 "test": N_samples_test}

    for subset in ["train", "dev", "test"]:
        print("------------- ", subset, "---------------")
        df_outputs[raio_name+subset] = df_outputs[met+subset] / np.log10(df_outputs["k_features"])
        df_outputs["diff_"+raio_name + subset] = df_outputs[raio_name + subset] - df_outputs[raio_name + subset].shift(periods=1)
        df_outputs["diff_performance_" + subset] = df_outputs[met+subset] - df_outputs[met+subset].shift(periods=1)
        df_outputs["diff_performance_FEAT" + subset] = (df_outputs[met + subset] - df_outputs[met + subset].shift(periods=1))/(df_outputs["k_features"] - df_outputs["k_features"].shift(periods=1))
        print(df_outputs[["k_features", "diff_performance_FEAT" + subset,"diff_performance_" + subset,met+subset, raio_name+subset]])
        # for i, row in df_outputs.iterrows():
        #     aic = calculate_aic(N_samples[subset], row[raio_name+subset], row["k_features"])
        #     print(aic)






    create_plot_dims(df_outputs, x="k_features", y=[metric_name+"_train", metric_name+"_dev", metric_name+"_test"],
                     colors=["magenta", "orange", "purple"], label=["train", "dev", "test"],
                     title=DS + "_" + MODEL_PATH + "\n"+metric_name,
                     metric=metric_name, n_samples=[N_samples_train, N_samples_dev, N_samples_test])

