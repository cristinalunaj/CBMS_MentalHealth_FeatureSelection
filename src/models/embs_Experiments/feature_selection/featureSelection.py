"""
	author: Cristina Luna,

"""

import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
import argparse
import pandas as pd
from src.dataLoader.embsLoader import EmbsLoaderCounselChat
from src.models.embs_Experiments.FeatureTraining import *
from src.metrics.metrics_classification import get_eval_metrics, calculate_CI
import matplotlib.pyplot as plt


def multiassign(d, keys, values):
    for k, v in zip(keys, values):
        d[k] = v

def create_plot_dims(df, x, y=[], colors = [], label = [], title="", metric="", n_samples=[]):
    ax = plt.gca()
    for i in range(len(y)):
        #plt.plot(df[x], df[y[i]], 'o--', color=colors[i], label=label[i])
        y_err = []
        for score in df[y[i]]:
            y_err += [calculate_CI(score, n_samples[i], confidence=0.95)]
        plt.errorbar(df[x], df[y[i]], y_err, fmt='o--', color=colors[i], label=label[i])
    plt.grid(color='0.95')
    ax.set_ylim([0, 100])
    plt.legend()
    plt.xlabel("Number of Features")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # parse input arguments
    # Read input parameters
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-DS', '--dataset_name', type=str, required=True,
                        help='[7cups, counseilChat]', default='counseilChat')
    parser.add_argument('-model', '--model_name', type=str, required=True,
                        help='[meta-llama/Llama-2-7b-chat-hf]')
    parser.add_argument('-norm', '--type_norm', type=int, help='Path to save the embeddings extracted from the model',
                        default=2)
    parser.add_argument('-fs', '--feature_selection', type=int,
                        help='1:"ANOVA-F",',
                        default=1)
    args = parser.parse_args()

    ###################################
    ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection"
    # Load dataset:
    DS = args.dataset_name #"counseilChat"
    MODEL_PATH = args.model_name #
    splitName = "complete_all_TopicClassification.csv"  # "test/eval/train.csv"
    QAandTask = "_all_TopicClassification.csv"
    col_labels_nums, col_labels_text = "labels", "topic"

    ###### CLASSIFICATION PARAMS: ######
    type_of_norm=args.type_norm  #0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]
    featureSelection_num = args.feature_selection
    get_posteriors = False
    dict_featureSelection = {
        1: "ANOVA-F...k:",
    }

    if("word2Vec" in MODEL_PATH):
        list_features2eval = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800]
        n_embs = 768
    elif("llama" in MODEL_PATH.lower()):
        list_features2eval = [10,50,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4100]
        n_embs = 4096
    else:
        list_features2eval = [10]
        n_embs = 0


    print(">>> RUNNING: ", DS, " - ", MODEL_PATH , " - ", dict_featureSelection[featureSelection_num])
    print(" FEATURES: ", str(list_features2eval))

    dict_model_combinations = {
        "meta-llama/Llama-2-7b-chat-hf": {
            "counseilChat": {"model_number": 6, "param_model": 0.001},
            "7cups": {"model_number": 6, "param_model": 0.001},
        },
        "klyang/MentaLLaMA-chat-7B": {
            "counseilChat": {"model_number": 6, "param_model": 0.01},
            "7cups": {"model_number": 6, "param_model": 0.001},
        },
        "word2Vec": {
            "counseilChat": {"model_number": 10, "param_model": 30},
            "7cups": {"model_number": 1, "param_model": 1.0},
    },
    }
    model_number = dict_model_combinations[MODEL_PATH][DS]["model_number"]
    param_model = dict_model_combinations[MODEL_PATH][DS]["param_model"]


    dict_model_param = {
        1: "SVC-C:",
        2: "LOGISTIC REGRES-C:",
        3: "RIDGE CLASSIF.-alpha:",
        # 4:"PERCEPT.-alpha:",
        5: "NU SVC-nu:",
        6: "LINEAR SVC-C:",
        7: "KNN-n_neighbors:",
        8: "NEAREST CENTROID-:",
        9: "DECISSION TREE-min_samples_split:",
        10: "RANDOM FOREST-n_estimators:",
        11: "MLP-hidden_layer_sizes:"
    }

    output2check = "_poolNorm"  # "pooler_output"
    out_dir_posteriors=""
    extraName = "_norm"+str(type_of_norm)+"_"

    out_path_output_featureSelection = os.path.join(ROOT_PATH,
                                  "data/experiments/featureSelection",
                                  MODEL_PATH.replace("/", "_"), DS, dict_featureSelection[featureSelection_num].split("...")[0].strip())
    os.makedirs(out_path_output_featureSelection, exist_ok=True)
    root_path_embs = os.path.join(ROOT_PATH,
                                  "data/experiments/embsClusterPooling",
                                  MODEL_PATH.replace("/", "_"), DS, "embs", output2check)



    filterByCols = {}  # {"setName": "Train"} #Train / Dev / Test
    cols_embs = ['emb{}'.format(i) for i in range(n_embs)]
    df_outputs = pd.DataFrame([], columns=["dataset", "embs_from_model", "task", "norm", "classifier_config", "feature_selection_config", "featuresSelected",
                       "avg_accuracy_train", "weighted_f1_train", "micro_f1_train", "macro_f1_train", "precision_SC_train", "recall_SC_train",
                       "avg_accuracy_dev", "weighted_f1_dev", "micro_f1_dev", "macro_f1_dev", "precision_SC_dev", "recall_SC_dev",
                       "avg_accuracy_test", "weighted_f1_test", "micro_f1_test", "macro_f1_test","precision_SC_test", "recall_SC_test"])

    dict_Xy = {}
    for splitName in ["Train", "Dev", "Test"]:
        labels_path_set = os.path.join(ROOT_PATH,
                                         "data/" + DS + "/dataSplits/" + splitName.lower()+QAandTask)
        filterByCols = {"setName": splitName}
        # LOAD EMBEDDINGS OF Q/A
        embsLoader_obj = EmbsLoaderCounselChat(root_path_embs, labels_path_set, col_setName="setName",
                                               filterBycol=filterByCols)
        complete_embs_set, _ = embsLoader_obj.load_embs(n_embs=n_embs)
        df_labels = embsLoader_obj.get_labels(labels_cols=["topic", "labels", "QorA"])
        df_merged_embs_labels = complete_embs_set.merge(df_labels, on="idx_set")
        df_merged_embs_labels["QorA_number"] = df_merged_embs_labels['QorA'].apply(lambda a: 1 if a == 'Q' else 0)

        print(">> N samples of "+ splitName+":", str(len(df_merged_embs_labels)))
        dict_Xy[splitName] = df_merged_embs_labels

    X_train = dict_Xy["Train"][cols_embs]
    y_train = dict_Xy["Train"]["topic"] #y_train.astype('category').cat.codes


    X_dev = dict_Xy["Dev"][cols_embs]
    y_dev = dict_Xy["Dev"]["topic"]

    X_test = dict_Xy["Test"][cols_embs]
    y_test = dict_Xy["Test"]["topic"]

    if(os.path.exists(os.path.join(out_path_output_featureSelection, QAandTask + extraName + "out_FS_simpleClassifiers.csv"))):
        # Load
        df_outputs = pd.read_csv(os.path.join(out_path_output_featureSelection, QAandTask + extraName + "out_FS_simpleClassifiers.csv"), sep=";", header=0)
    else:
        for param_fs in list_features2eval:
            print( ">>> N PARAMS: ", str(param_fs), " <<<")
            ########### FEATURE SELECTION ###############
            if(param_fs<len(cols_embs)):
                fs = feature_selection(X_train, y_train, featureSelection_num, out_path_output_featureSelection,param=param_fs)
                # apply feature selection
                fs.fit(X_train, y_train)
                # Transform data
                X_train_fs = fs.transform(X_train)
                X_dev_fs = fs.transform(X_dev)
                X_test_fs = fs.transform(X_test)
                if (featureSelection_num == 1):
                    names_Selected_features = list(fs.get_feature_names_out())
                    print("Selected features: ", names_Selected_features)

            else:
                X_train_fs = X_train
                X_dev_fs = X_dev
                X_test_fs = X_test



            ########### TRAINING ###############
            # Train classifier:
            print("Evaluating:", dict_model_param[model_number] + str(param_model))
            aux_dict_tests = {}
            multiassign(aux_dict_tests,
                        ["dataset", "embs_from_model", "task", "norm", "classifier_config", "feature_selection_config", "k_features","featuresSelected"],
                        [DS, MODEL_PATH, QAandTask, type_of_norm, dict_model_param[model_number] + str(param_model),
                         dict_featureSelection[featureSelection_num] + str(param_fs), (param_fs), [names_Selected_features]])


            # Classification:
            classifier = get_classifier(model_number, param_model, seed=2020)
            classifier.fit(X_train_fs, y_train)


            # METRICS:
            print(" ------------ TRAIN METRICS ---------------")
            predictions_train = classifier.predict(X_train_fs)
            multiassign(aux_dict_tests,
                        ["avg_accuracy_train", "weighted_f1_train", "micro_f1_train", "macro_f1_train", "precision_SC_train",
                         "recall_SC_train"], list(get_eval_metrics(predictions_train, y_train)))
            print(" ------------ DEV METRICS ---------------")
            predictions_dev = classifier.predict(X_dev_fs)
            multiassign(aux_dict_tests,
                        ["avg_accuracy_dev", "weighted_f1_dev", "micro_f1_dev", "macro_f1_dev", "precision_SC_dev",
                         "recall_SC_dev"], list(get_eval_metrics(predictions_dev, y_dev)))
            print(" ------------ TEST METRICS ---------------")
            predictions_test = classifier.predict(X_test_fs)
            multiassign(aux_dict_tests,
                        ["avg_accuracy_test", "weighted_f1_test", "micro_f1_test", "macro_f1_test", "precision_SC_test",
                         "recall_SC_test"], list(get_eval_metrics(predictions_test, y_test)))

            ## Add aux dict to datafraem:
            df_outputs = pd.concat([df_outputs, pd.DataFrame([aux_dict_tests.values()], columns=list(aux_dict_tests.keys()))],
                                   ignore_index=True)
            # Save dataframe:
            df_outputs.to_csv(
                os.path.join(out_path_output_featureSelection, QAandTask + extraName + "out_FS_simpleClassifiers.csv"),
                sep=";", header=True, index=False)
            print(" RESULTS SAVED IN: ",
                  os.path.join(out_path_output_featureSelection,
                               QAandTask + extraName + "out_FS_simpleClassifiers.csv"))



    ### PLOT METRICS #############################333
    create_plot_dims(df_outputs, x="k_features", y=["weighted_f1_train", "weighted_f1_dev", "weighted_f1_test"],
                     colors=["magenta", "orange", "purple"], label=["train", "dev", "test"],
                     title=DS + "_" + MODEL_PATH + "\nWeighted-F1 - " + dict_model_param[model_number] + str(
                         param_model) + " -- " + dict_featureSelection[featureSelection_num].split("...")[0],
                     metric="Weighted-F1", n_samples=[len(X_train), len(X_dev), len(X_test)])
    create_plot_dims(df_outputs, x="k_features", y=["macro_f1_train", "macro_f1_dev", "macro_f1_test"],
                     colors=["magenta", "orange", "purple"], label=["train", "dev", "test"],
                     title=DS + "_" + MODEL_PATH + "\nMacro-F1 - " + dict_model_param[model_number] + str(
                         param_model) + " -- " + dict_featureSelection[featureSelection_num].split("...")[0],
                     metric="Macro-F1", n_samples=[len(X_train), len(X_dev), len(X_test)])



