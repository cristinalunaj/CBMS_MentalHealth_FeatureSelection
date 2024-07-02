# CBMS_MentalHealth_FeatureSelection
Repository of the paper presented at CBMS2024 conference: Mental-Health Topic Classification employing D-vectors of Large Language Models



## Install requirements/packages:

Python 3.10.12

To install the python packages, create a new virtual environment and run:
    
    pip install -r requirements.txt

** If problems installing certain libraries, try to update your pip version: pip3 install --upgrade pip and run again the previous command

## Datasets 


### 1) Counseil-Chat


Dataset released under MIT license, if you use this dataset, cite it as:

@misc{bertagnolli2020counsel,
  title={Counsel chat: Bootstrapping high-quality therapy data},
  author={Bertagnolli, Nicolas},
  year={2020},
  publisher={Towards Data Science. https://towardsdatascience. com/counsel-chat~…}
}
More information in: (https://huggingface.co/datasets/nbertagnolli/counsel-chat)
* In CBMS_MentalHealth_FeatureSelection/data/counseilChat/dataSplits/complete_all_TopicClassification.csv > We release the splits used in the experiments of the current publication but if you use
the dataset you MUST cite the original source posted before. 



### 2) 7Cups

Question and answers from: https://www.7cups.com/qa/
(In process of cleaning and evaluating for being released)


## 1. Datasets Splits Extraction
The code for generating the splits of the dataset is available in: 

    src/DSprocessor/TopicChatProcessor.py
Remember to change the parameter, before running it: 

    ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection"

In case of problems, re-use the data-splits that appear in:
    
    data/counseilChat/dataSplits/complete_all_TopicClassification.csv 



## 2. Baseline

### 2.1. Word2Vec
- Select the dataset to use: e.g. (DS = "counseilChat") 
- Select the model to use: e.g (MODEL_PATH = 'word2Vec')
- Select the number of features (dimension of the embeddings) to use with Word2Vec: n_embs = 768


    python3 CBMS_MentalHealth_FeatureSelection/src/models/embs_Experiments/word2Vec.py

## 3. EXTRACT EMBEDDINGS 


### 3.1. LLAMA FAMILY EMBEDDINGS

Go to src/embs_Experiments/embs_Llama.py

- Select the dataset to use: e.g. (DS = "counseilChat") 
- Select the model to use: e.g (MODEL_PATH = 'meta-llama/Llama-2-7b-chat-hf')
- Add your own access_token from HuggingFace library (if the model requires it) 
- Optionally, you can also modify other parameters to change the prompt (prompt_template), maximum number of tokens (max_num_rokens), plus the configuration to load the LLM: 
  - bitesAndBytes_config = {
          "load_in_4bit": True,
          "bnb_4bit_use_double_quant": False,
          "bnb_4bit_quant_type": "nf4",
          "bnb_4bit_compute_dtype": torch.bfloat16
      }

## 4. Simple model classifier
Once embeddings are generated, go to src/models/classification.py

- Select the dataset to use: e.g. (DS = "counseilChat") 
- Select the model from which we want to use the embeddings to train the classifiers: e.g (MODEL_PATH = 'meta-llama/Llama-2-7b-chat-hf')
- Select the list of models to train. Keys indicate the model from dict_model_param, and the list of values, the hyperparameters to eval for each
of the parameter selected in each case (e.g. 1: [0.001], indicates to train an SVC model from sklearn with C=0.001).
  - list_models_params = {1:[0.001,0.01,0.1,1.0,10,100],
                          2:[0.001,0.01,0.1,1.0,10,100],
                          #5:[0.2,0.4,0.5,0.6,0.8,1.0],
                          6:[0.001,0.01,0.1,1.0,10,100],
                          7:[5,10,15,20,25,30],
                          8:[0],
                          9:[5,10,15,20,25,30],
                          10:[5,10,15,20,25,30,40,50,60,70,80,90],
                          11: ['(80)', '(80,80)', '(80,80,80)'],
                          }

  


## 5. Feature Selection & Eval classifiers

Perform feature selection & training of the top classifiers with different number of features:

    python3 src/models/embs_Experiments/feature_selection/featureSelection.py 
    --dataset_name counseilChat --model_name meta-llama/Llama-2-7b-chat-hf --feature_selection 1


Plot graphics for comparing models:

    python3 src/models/embs_Experiments/feature_selection/top_features_study.py
    
    python3 src/models/embs_Experiments/feature_selection/top_ratio_results.py




## FAQ

* If you experiment problems with the path, try to change the ROOT_PATH parameter to an absolute path (e.g. ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection" to ROOT_PATH = "/home/xx/CBMS_MentalHealth_FeatureSelection")
* NLP library requires to install certain extra packages (nltk.install (...) read the error messages and follow the instructions)


## Citation:
TO ADD ONCE PUBLISHED WITH THE REST OF THE CODE



### License:
Apache 2.0.










