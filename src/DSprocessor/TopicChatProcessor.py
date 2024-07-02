"""
	author: Cristina Luna

"""
import os
import pandas as pd
import torch

##DATASET
from datasets import load_dataset


##### SET RANDOMIZERS
from src.utils.fix_randomness import seed_torch
import random


def check_statistics_in_set(df_complete):
    print("> Total length: ", len(df_complete))
    print("> Number of different therapists: ", len(df_complete["therapistURL"].unique()))
    print("> Number of different topics: ", len(df_complete["topic"].unique()))
    print("TOPICS: ", df_complete["topic"].value_counts())
    print("> Number of different questions: ", len(df_complete["questionText"].unique()))
    #print("> Indexes: ", list(df_complete["idx"]))


def save_ds(save_path_ds, df_set, splitname = "train", data2useAsText = "all", task=""):
    if (save_path_ds != ""):
        os.makedirs(save_path_ds, exist_ok=True)
        # save dataset
        df_set.to_csv(os.path.join(save_path_ds, splitname +"_" +data2useAsText +task+".csv"), sep=";", index=False, header=True)

def load_ds(save_path_ds, splitname = "train", data2useAsText = "all", task=""):
    df_set = pd.read_csv(os.path.join(save_path_ds, splitname +"_" +data2useAsText +task+".csv"), sep=";", header=0)
    return df_set




def prepare_subset(df, dict_topics, data2useAsText='all', labels_col = "topic", columnswithNonAllowedDuplicates=["answerText", "questionText"]):
    # Remove duplicated answers if they exist
    #dropped_data = df.loc[df.duplicated(subset=["answerText", "questionText"])]
    df = df.drop_duplicates(subset=columnswithNonAllowedDuplicates)
    # Select the text to train the topic-classificator (only questions, only answers, or both=all)
    if (data2useAsText == "onlyQuestions"):
        # Drop suplicate questions:
        df["text"] = df["questionText"]
        df["QorA"] = "Q" # marker to differenciate/filter lates questions from answers
    elif (data2useAsText == "onlyAnswers"):
        df["text"] = df["answerText"]
        df["QorA"] = "A" # marker to differenciate/filter lates questions from answers
    else:
        df["QorA"] = "A" # marker to differenciate/filter lates questions from answers
        # Get text of questions & add to the column of answerText (because we are going to maintain it):
        train_df_auxQuestions = df.drop_duplicates(subset=["questionText"])
        train_df_auxQuestions["answerText"] = train_df_auxQuestions["questionText"]
        train_df_auxQuestions["QorA"] = "Q" # marker to differenciate/filter lates questions from answers
        # Add additional rows with the questions in the 'text' column:
        df = pd.concat([df, train_df_auxQuestions])
        df["text"] = df["answerText"]
    # Remove duplicates in text if they exist
    df = df.drop_duplicates(subset=["text"])
    # Clean text (HTML comments)
    #df["text"] = df["text"].apply(lambda x: clean_CounseilChat(x))
    df["labels"] = df[labels_col].apply(lambda x: dict_topics[x])
    df = df.reset_index(drop=True)
    check_statistics_in_set(df)
    df = df.sample(frac=1)
    return df


def get_dicts_DS(DS):
    if(DS=="counseilChat"):
        dict_topics = {  ## 30 topics
            'depression': 0,
            'anxiety': 1,
            'counseling-fundamentals': 2,
            'intimacy': 3,
            'relationships': 4,
            'parenting': 5,
            'family-conflict': 6,
            'trauma': 7,
            'relationship-dissolution': 8,
            'self-esteem': 9,
            'behavioral-change': 10,
            'marriage': 11,
            'lgbtq': 12,
            'anger-management': 13,
            'spirituality': 14,
            'substance-abuse': 15,
            'workplace-relationships': 16,
            'professional-ethics': 17,
            'grief-and-loss': 18,
            'social-relationships': 19,
            'diagnosis': 20,
            'domestic-violence': 21,
            'eating-disorders': 22,
            'sleep-improvement': 23,
            'addiction': 24,
            'legal-regulatory': 25,
            'human-sexuality': 26,
            'children-adolescents': 27,  ########
            'stress': 28,  #########
            'military-issues': 29      ######
        }
    elif(DS=="7cups"):
        dict_topics = {
            'Relationship':0,
            'Chronic':1,
            'Work':2,
            'Managing':3,
            'Recovery':4,
            'Domestic':5,
            'Alcohol-Drug':6,
            'Parenting':7,
            'Obsessive':8,
            'Anxiety':9,
            'Borderline':10,
            'Getting':11,
            'Grief':12,
            "Women's":13,
            'Autism':14,
            'Spirituality':15,
            'Social':16,
            'Exercise':17,
            'Bullying':18,
            'Sexual':19,
            'Depression':20,
            'LGBTQ+':21,
            'Weight':22,
            'Panic':23,
            'Student':24,
            'Sleeping':25,
            'Family':26,
            'Financial':27,
            'Breakups':28,
            'Self-Esteem':29,
            'Loneliness':30,
            'Bipolar':31,
            'General':32,
            'PTSD' :33,
            'Eating':34,
            'Self-Harm':35,
            'ADHD':36,
            'Disabilities':37,
            'Forgiveness':38
        }
    else:
        print("to do")
        dict_topics = {}

    return dict_topics


def create_subsets_TopicClassification(dataset, save_path_ds, data2useAsText = "all",
                                                    splits=[0.8, 0.1, 0.1], minQuestPerTopic=3, minSamplesPerTopic=-1,
                                                    datasetName="counseilChat"):
    """
    Divide the dataset to have only one tipe of Q/A per set (not having answers or questions in different sets) but with
    all the topics in each set (or as many as possible) since the task is topic classification and we need to have
    representative populations of the class in all the sets.

    """
    task = "_TopicClassification"
    dict_topics = get_dicts_DS(datasetName)

    # Check if exist sets:
    if(os.path.exists(os.path.join(save_path_ds ,"train" +"_" +data2useAsText +task+".csv"))):
        # Load dataset:
        print("Data was already generated! Loading from local folder: ", os.path.join(save_path_ds ,"xxx" +"_" +data2useAsText +task+".csv"))
        train_df = load_ds(save_path_ds, splitname="train", data2useAsText=data2useAsText, task=task)
        print("# Samples training: ", str(len(train_df)))
        eval_df = load_ds(save_path_ds, splitname="eval", data2useAsText=data2useAsText, task=task)
        print("# Samples eval: ", str(len(eval_df)))
        test_df = load_ds(save_path_ds, splitname="test", data2useAsText=data2useAsText, task=task)
        print("# Samples test: ", str(len(test_df)))
        complete_df = load_ds(save_path_ds, splitname="complete", data2useAsText=data2useAsText, task=task)
        print("# Samples complete: ", str(len(complete_df)))
        check_statistics_in_set(complete_df)
        dataset_sets = {"train": train_df,
                        "eval": eval_df,
                        "test": test_df,
                        "complete": complete_df}
    else: # create splits:
        # Split by topic
        df_complete = dataset
        check_statistics_in_set(df_complete)
        # Discard duplicate answers (WE GIVE UP TO ONE QUESTIONS WITH THE AIM OF HAVING NON-REPEATED TEXT IN ANY SET (TRAIN/VAL/TEST):
        df_complete = df_complete.drop_duplicates(subset=["answerText"])

        print("--------------STATISTICS INITIAL 'CLEANED' DATASET: --------------")
        check_statistics_in_set(df_complete)
        print("------------------------------------------------------------------")

        Nsamples_train, Nsamples_dev, Nsamples_test = int(len(df_complete) * splits[0]), int(
            len(df_complete) * splits[1]), int(len(df_complete) * splits[2])
        print("# Train ", str(len(df_complete) * splits[0]))
        print("# Eval ", str(len(df_complete) * splits[1]))
        print("# Test ", str(len(df_complete) * splits[2]))
        train_df = pd.DataFrame([], columns=df_complete.columns)
        eval_df = pd.DataFrame([], columns=df_complete.columns)
        test_df = pd.DataFrame([], columns=df_complete.columns)
        # df_complete_check = prepare_subset(df_complete, dict_topics, data2useAsText=data2useAsText)
        # check_statistics_in_set(df_complete_check)
        ######
        # Create sets:
        # ORDER BY TOPIC:
        groups_topics = df_complete.groupby(["topic"])
        for i, group_topic_df in groups_topics:
            groups_QID = group_topic_df.groupby(["questionID"])
            N_questions = len(groups_QID)
            N_samples = len(group_topic_df)
            if(N_questions<minQuestPerTopic or N_samples<minSamplesPerTopic):
                # Discard topic since we do not have enough samples to distribute in sets
                print("Discarding topic: ", group_topic_df["topic"].unique()[0])
                print("Discarding samples: ", len(group_topic_df))
            else:
                # Distribute in groups of questionID:
                for j, group_qID_df in groups_QID:
                    # Fill in at least 1 question of each topic per set:
                    if(len(train_df.loc[train_df["topic"]==i[0]])<=0):
                        # add first value:
                        train_df = train_df._append(group_qID_df)
                    elif(len(eval_df.loc[eval_df["topic"]==i[0]])<=0):
                        eval_df = eval_df._append(group_qID_df)
                    elif(len(test_df.loc[test_df["topic"]==i[0]])<=0):
                        test_df = test_df._append(group_qID_df)
                    else: # For the rest of samples, Add samples according to expeceted distribution:
                        random_sample = random.uniform(0, 1)
                        if ((random_sample < splits[0]) & (len(train_df) < Nsamples_train)):
                            train_df = train_df._append(group_qID_df)
                        elif ((random_sample < (splits[0] + splits[1])) & (len(eval_df) < Nsamples_dev)):
                            eval_df = eval_df._append(group_qID_df)
                        elif (random_sample <= (splits[0] + splits[1] + splits[2])):
                            test_df = test_df._append(group_qID_df)
        # Replace topics by value in dictionary:
        print(">>>>>>>>>>> Train:")
        # reset index:
        train_df = prepare_subset(train_df, dict_topics, data2useAsText=data2useAsText)
        train_df["idx_set"] = list(range(0, len(train_df)))
        train_df["setName"] = "Train"
        print(">>>>>>>>>>> Dev:")
        eval_df = prepare_subset(eval_df, dict_topics, data2useAsText=data2useAsText)
        eval_df["idx_set"] = list(range(len(train_df), len(train_df)+len(eval_df)))
        eval_df["setName"] = "Dev"
        print(">>>>>>>>>>> Test:")
        test_df = prepare_subset(test_df, dict_topics, data2useAsText=data2useAsText)
        test_df["idx_set"] = list(range(len(train_df)+len(eval_df), len(train_df)+len(eval_df)+len(test_df)))
        test_df["setName"] = "Test"
        print("Complete:")
        compl_df = pd.concat([train_df, eval_df, test_df]) #prepare_subset(df_complete, dict_topics, data2useAsText=data2useAsText)
        check_statistics_in_set(compl_df)
        # Save new sets:
        save_ds(save_path_ds, train_df, splitname="train", data2useAsText=data2useAsText, task=task)
        save_ds(save_path_ds, eval_df, splitname="eval", data2useAsText=data2useAsText, task=task)
        save_ds(save_path_ds, test_df, splitname="test", data2useAsText=data2useAsText, task=task)
        save_ds(save_path_ds, compl_df, splitname="complete", data2useAsText=data2useAsText, task=task)
        dataset_sets = {"train": train_df,
                        "eval": eval_df,
                        "test": test_df,
                        "complete": compl_df}
    return dataset_sets


if __name__ == '__main__':
    #### INITIAL CONFIGURATIONS #####
    seed_id = 2020
    seed_torch(seed=seed_id)  # Set random seed
    ############################################
    ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection"
    # Load dataset:
    DS = "counseilChat"  # counseilChat    7cups
    data2useAsText = "all" # all onlyQuestions onlyAnswers
    task = "_TopicClassification" # _QA "_TopicClassification"
    splitName = "test_all.csv"  # "test/eval/train.csv"
    save_path_ds = os.path.join(ROOT_PATH,"data/" + DS + "/dataSplits")
    os.makedirs(save_path_ds, exist_ok=True)

    # CREATE DATASET SPLITS:
    if (DS == "counseilChat"):
        ## LOAD & PREPARE DATASET:
        if (not os.path.exists(os.path.join(save_path_ds ,"train " +"_" +data2useAsText + task+".csv"))):
            # Load dataset:
            dataset = load_dataset("nbertagnolli/counsel-chat")
            print("Initial number of samples: ", str(dataset.shape["train"][0]))
            # Add index
            dataset["train"] = dataset["train"].add_column("idx", list(range(0, dataset.shape["train"][0])))
            # Remove files with empty questions or answers:
            cleaned_DS = dataset.filter(lambda sample: (sample["questionText"] != None and sample["answerText"] != None))
            cleaned_DS = pd.DataFrame(cleaned_DS["train"])
            print("Number of samples after cleaning: ", str(len(cleaned_DS)))
            complete_df = pd.DataFrame(cleaned_DS["train"])

    elif(DS=="7cups"):
        ## LOAD & PREPARE DATASET:
        if (not os.path.exists(os.path.join(save_path_ds ,"train " +"_" +data2useAsText + task+".csv"))):
            # Load dataset:
            dataset_path = os.path.join(ROOT_PATH, "data/7cups/complete_df.csv")
            dataset = pd.read_csv(dataset_path, sep=";", header=0)
            print("Initial number of samples: ", str(len(dataset)))
            # Add index
            dataset["idx"] = list(range(0, len(dataset)))
            # Change name of columns to match counsel-chat DS columns
            dataset.rename(columns={"Question_Txt": "questionText",
                                    "answer_Txt": "answerText",
                                    "Question_category": "topic",
                                    "Question_ID": "questionID",
                                    "moderatedBy_URL": "therapistURL"}, inplace=True)

            # Remove files with empty questions or answers:
            cleaned_DS = dataset.loc[(dataset["questionText"] != None) & (dataset["answerText"] != None)]
            complete_df = cleaned_DS
    else:
        print("to do")
        cleaned_DS = []

    # PREPARE DATASET: (if data was already generated

    if (task == "_TopicClassification"):
        dataset_sets = create_subsets_TopicClassification(complete_df, save_path_ds, splits=[0.8, 0.1, 0.1],
                                                          datasetName=DS)





