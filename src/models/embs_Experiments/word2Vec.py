"""
	author: Cristina Luna,

"""
import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
import numpy as np


def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def vectorize(sentence, n_embs=100):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(n_embs)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

if __name__ == '__main__':
    #### INITIAL CONFIGURATIONS #####
    seed_id = 2020

    ############################################
    ############################################
    ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection"
    # Load dataset:
    DS = "counseilChat" # counseilChat 7cups
    splitName = "complete_all_TopicClassification.csv"  # "test/eval/train.csv"
    MODEL_PATH = 'word2Vec'
    n_embs = 768
    EXTRA_NAME = ""
    output2check = "_poolNorm"
    path_dataset = os.path.join(ROOT_PATH, "data/" + DS + "/dataSplits/" + splitName)

    # save path:
    root_path_embs = os.path.join(ROOT_PATH, "data/experiments/embsClusterPooling",MODEL_PATH.replace("/", "_"), DS,
                                  "embs", output2check)

    os.makedirs(root_path_embs, exist_ok=True)


    df_complete = pd.read_csv(path_dataset, sep=";", header=0)

    # Divide into sets:
    df_train = df_complete.loc[df_complete["setName"]=="Train"]
    df_dev = df_complete.loc[df_complete["setName"] == "Dev"]
    df_test = df_complete.loc[df_complete["setName"] == "Test"]



    ### extract embessings per topic (keywords) ###
    X_train = df_train["text"].apply(preprocess)
    X_dev = df_dev["text"].apply(preprocess)
    X_test = df_test["text"].apply(preprocess)

    #Train a Word2Vec model on the preprocessed training data using Gensim package.
    # Check if model saved:
    if(os.path.exists(os.path.join(root_path_embs, "word2vec"+str(n_embs)+".model"))):
        # Load
        w2v_model = Word2Vec.load(os.path.join(root_path_embs, "word2vec"+str(n_embs)+".model"))
    else:
        sentences_train = [sentence.split() for sentence in X_train]
        w2v_model = Word2Vec(sentences_train, vector_size = n_embs, window=5, min_count=5, workers=4)
        # save
        w2v_model.save(os.path.join(root_path_embs, "word2vec"+str(n_embs)+".model"))


    # Convert the preprocessed text data to a vector representation using the Word2Vec model.
    X_train = np.array([vectorize(sentence, n_embs=n_embs) for sentence in X_train])
    X_dev = np.array([vectorize(sentence, n_embs=n_embs) for sentence in X_dev])
    X_test = np.array([vectorize(sentence, n_embs=n_embs) for sentence in X_test])

    # Save embeddings:
    cols_embs = ['emb{}'.format(i) for i in range(n_embs)]
    # Convert into dataframes:
    X_df_train = pd.DataFrame(X_train, columns=cols_embs)
    X_df_train["idx_set"] = list(df_train["idx_set"])
    X_embs_train = X_df_train.merge(df_train, on="idx_set")
    X_embs_train = X_embs_train.reset_index(drop=True, inplace=False)

    X_df_dev = pd.DataFrame(X_dev, columns=cols_embs)
    X_df_dev["idx_set"] = list(df_dev["idx_set"])
    X_embs_dev = X_df_dev.merge(df_dev, on="idx_set")
    X_embs_dev = X_embs_dev.reset_index(drop=True, inplace=False)

    X_df_test = pd.DataFrame(X_test, columns=cols_embs)
    X_df_test["idx_set"] = list(df_test["idx_set"])
    X_embs_test = X_df_test.merge(df_test, on="idx_set")
    X_embs_test = X_embs_test.reset_index(drop=True, inplace=False)

    X_embs_complete = pd.concat([X_embs_train, X_embs_dev, X_embs_test])
    X_embs_complete = X_embs_complete.reset_index(drop=True,inplace=False)
    # Save
    X_embs_complete.to_csv(os.path.join(root_path_embs, "Complete_avg_complete_embs.csv"), sep=";", index=False, header=True)
    X_embs_train.to_csv(os.path.join(root_path_embs, "Train_avg_complete_embs.csv"), sep=";", index=False,header=True)
    X_embs_dev.to_csv(os.path.join(root_path_embs, "Dev_avg_complete_embs.csv"), sep=";", index=False,header=True)
    X_embs_test.to_csv(os.path.join(root_path_embs, "Test_avg_complete_embs.csv"), sep=";", index=False,header=True)








