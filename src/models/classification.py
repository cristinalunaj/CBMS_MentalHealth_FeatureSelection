import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import argparse
import pandas as pd
from src.utils.fix_randomness import seed_torch
from src.dataLoader.embsLoader import EmbsLoaderCounselChat
from src.models.embs_Experiments.FeatureTraining import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.metrics.metrics_classification import get_eval_metrics
import pickle



def multiassign(d, keys, values):
    for k, v in zip(keys, values):
        d[k] = v

def get_dictWithModels():
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
    return dict_model_param



if __name__ == '__main__':
    # parse input arguments
    seed_id = 2020


    ###### CLASSIFICATION PARAMS: ######
    type_of_norm=2 #0-MinMax Norm / 1-Standard Norm / 2- No apply normalization [default: 2]
    param = 1.0
    get_posteriors = False
    dict_model_param = {
        1:"SVC-C:",
        2:"LOGISTIC REGRES-C:",
        3:"RIDGE CLASSIF.-alpha:",
        #4:"PERCEPT.-alpha:",
        5:"NU SVC-nu:",
        6:"LINEAR SVC-C:",
        7:"KNN-n_neighbors:",
        8:"NEAREST CENTROID-:",
        9:"DECISSION TREE-min_samples_split:",
        10:"RANDOM FOREST-n_estimators:",
        11: "MLP-hidden_layer_sizes:"
                        }
    list_models_params = {1:[0.001,0.01,0.1,1.0,10,100],
                          2:[0.001,0.01,0.1,1.0,10,100],
                          #5:[0.2,0.4,0.5,0.6,0.8,1.0],
                          6:[0.001,0.01,0.1,1.0,10],
                          7:[5,10,15,20,25,30],
                          8:[0],
                          9:[5,10,15,20,25,30],
                          10:[5,10,15,20,25,30,40,50,60,70,80,90],
                          11: ['(80)', '(80,80)', '(80,80,80)'],
                          }
    ###################################
    ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection"
    # Load dataset:
    DS = "counseilChat" #counseilChat 7cups
    MODEL_PATH = "klyang_MentaLLaMA-chat-7B word2Vec"  #  klyang_MentaLLaMA-chat-7B word2Vec
    splitName = "complete_all_TopicClassification.csv"  # "test/eval/train.csv"
    QAandTask = "_all_TopicClassification.csv"
    col_labels_nums, col_labels_text = "labels", "topic"

    if (MODEL_PATH=="word2Vec"):
        n_embs = 768
    else:
        n_embs = 4096
    output2check = "_poolNorm"  # "pooler_output"

    extraName = "_norm"+str(type_of_norm)+"_"
    out_path_output_classification = os.path.join(ROOT_PATH,
                                  "data/experiments/simpleClassifiers",
                                  MODEL_PATH.replace("/", "_"), DS, "embs", output2check)
    out_dir_posteriors = os.path.join(ROOT_PATH,
                                  "data/experiments/simpleClassifiers",
                                  MODEL_PATH.replace("/", "_"), DS, "posteriors")

    os.makedirs(out_path_output_classification, exist_ok=True)
    os.makedirs(out_dir_posteriors, exist_ok=True)
    root_path_embs = os.path.join(ROOT_PATH,
                                  "data/experiments/embsClusterPooling",
                                  MODEL_PATH.replace("/", "_"), DS, "embs", output2check)
    root_save_path = os.path.join(ROOT_PATH, "data/experiments/Cluster",
                                  MODEL_PATH.replace("/", "_"), DS, "DISTANCES_results")

    filterByCols = {}  # {"setName": "Train"} #Train / Dev / Test
    cols_embs = ['emb{}'.format(i) for i in range(n_embs)]
    df_outputs = pd.DataFrame([], columns=["dataset", "embs_from_model", "task", "norm", "classifier_config",
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


    for model_number in list(list_models_params.keys()):
        for param in list_models_params[model_number]:
            classifier_name = dict_model_param[model_number] + str(param).strip()
            print("Evaluating:", classifier_name)
            aux_dict_tests = {}
            multiassign(aux_dict_tests,
                        ["dataset", "embs_from_model", "task", "norm", "classifier_config"], [DS,MODEL_PATH, QAandTask, type_of_norm, classifier_name])

            X_train = dict_Xy["Train"][cols_embs]
            y_train = dict_Xy["Train"]["topic"]

            X_dev = dict_Xy["Dev"][cols_embs]
            y_dev = dict_Xy["Dev"]["topic"]

            X_test = dict_Xy["Test"][cols_embs]
            y_test = dict_Xy["Test"]["topic"]

            # Normalize (if required)
            if (int(type_of_norm) in [0, 1]):
                if (type_of_norm == 1):
                    scaler = MinMaxScaler(feature_range=(0, 1))
                else:
                    scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_dev = scaler.transform(X_dev)
                X_test = scaler.transform(X_test)

            # Classification:
            filename_classifier = os.path.join(out_path_output_classification,
                                               classifier_name.replace(":", "--") + ".sav")
            if(os.path.exists(filename_classifier)):
                print("Loading classifier")
                classifier = pickle.load(open(filename_classifier, 'rb'))
                print(classifier.classes_)
            else:
                print("Training classifier")
                classifier = get_classifier(model_number, param, seed=2020, get_posteriors=get_posteriors)
                classifier.fit(X_train, y_train)
            dict_classes_names = dict(zip(classifier.classes_, range(0, len(classifier.classes_))))
            inv_dict_classes_names = dict(zip(range(0, len(classifier.classes_)), classifier.classes_))

            # METRICS:
            print(" ------------ TRAIN METRICS ---------------")
            predictions_train = classifier.predict(X_train)
            multiassign(aux_dict_tests, ["avg_accuracy_train", "weighted_f1_train", "micro_f1_train", "macro_f1_train", "precision_SC_train", "recall_SC_train"], list(get_eval_metrics(predictions_train, y_train)))
            print(" ------------ DEV METRICS ---------------")
            predictions_dev = classifier.predict(X_dev)
            multiassign(aux_dict_tests,["avg_accuracy_dev", "weighted_f1_dev", "micro_f1_dev", "macro_f1_dev", "precision_SC_dev", "recall_SC_dev"], list(get_eval_metrics(predictions_dev, y_dev)))
            print(" ------------ TEST METRICS ---------------")
            predictions_test = classifier.predict(X_test)
            multiassign(aux_dict_tests,["avg_accuracy_test", "weighted_f1_test", "micro_f1_test", "macro_f1_test","precision_SC_test", "recall_SC_test"], list(get_eval_metrics(predictions_test, y_test)))

            ## Add aux dict to datafraem:
            df_outputs = pd.concat([df_outputs, pd.DataFrame([aux_dict_tests.values()], columns=list(aux_dict_tests.keys()))], ignore_index=True)
    # Save dataframe:
    df_outputs.to_csv(os.path.join(out_path_output_classification,QAandTask+extraName+"out_simpleClassifiers.csv"), sep=";", header=True, index=False)
    print(" RESULTS SAVED IN: ", os.path.join(out_path_output_classification,QAandTask+extraName+"out_simpleClassifiers.csv"))

