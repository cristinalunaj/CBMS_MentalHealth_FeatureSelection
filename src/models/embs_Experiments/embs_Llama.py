"""
	author: Cristina Luna,

"""
import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')
import torch
import pandas as pd
from src.DSprocessor.TopicChatProcessor import get_dicts_DS
from src.utils.fix_randomness import seed_torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizerFast
import numpy as np
from src.utils.tokens_management import remove_words, cleanSentence




def generate_embs_Llama2(df_test, model, tokenizer, device, max_num_tokens=1024, column2extractText="text",
                      root_save_path="", prompt_template = '"<DATA2INSERT>"'):
    output_i = "_poolNorm"
    idx_seq = 0
    for i, row in df_test.iterrows():
        idx = row["idx_set"]
        out_path_embs = os.path.join(root_save_path, "embs", output_i, str(idx))

        if (not os.path.exists(out_path_embs)):

            qa = row[column2extractText]  # questionText  answerText
            qa = cleanSentence(qa)

            # Count number of tokens of the Q/A:
            # Count tokens ---------------
            initial_tokens = tokenizer(qa, return_tensors="pt")
            n_tokens = initial_tokens.data["input_ids"].detach().shape[-1]
            print("Initial N tokens: ", n_tokens)
            if (n_tokens > max_num_tokens):  # max_length
                # 100 tokens ~ 75 words (3/4)
                # Truncate sentence (maintaining the end)
                qa = remove_words(qa, tokenizer, max_num_tokens=max_num_tokens)
            # ------------------------------

            prompt_qa = prompt_template.replace("<DATA2INSERT>", qa)
            inputs = tokenizer(prompt_qa, return_tensors="pt")
            out = model(**inputs, return_dict=True)
            sentence_embeddings = mean_pooling_timestamps(out["last_hidden_state"])

            # SAVE OUTPUT:
            ## SAVE BOTH OUTPUTS:
            if (root_save_path != ""):
                # for output_i in outputs2save:
                # out_path_embs = os.path.join(root_save_path, "embs", output_i, str(idx))
                os.makedirs(out_path_embs, exist_ok=True)
                out_path_emb_i = os.path.join(out_path_embs, str(idx) + "_part_" + str(idx_seq).zfill(2) + ".npy")
                with open(out_path_emb_i, "wb") as f:
                    if (not (device == "cpu")):
                        sentence_embeddings = sentence_embeddings.cpu()
                    np.save(f, sentence_embeddings.detach().numpy())  # output_model[output_i]
                    del sentence_embeddings
                # save csv:
                row["prompt_qa"] = prompt_qa
                pd.DataFrame([row], columns=list(row.index)).to_csv(
                    os.path.join(out_path_embs, str(idx) + "_part_" + str(idx_seq).zfill(2) + ".csv"), index=False,
                    sep=";", header=True)


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling_timestamps(model_output):
    return torch.mean(model_output, dim=1)





if __name__ == '__main__':
    #### INITIAL CONFIGURATIONS #####
    seed_id = 2020
    seed_torch(seed=seed_id)  # Set random seeds
    # CONFIG GPU TO USE:
    print("Is cuda available?", torch.cuda.is_available())
    print("Is cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled? ", torch.backends.cudnn.enabled)
    print("Device count?", torch.cuda.device_count())
    # Select GPU:
    gpu = ":0"
    device = torch.device("cuda"+gpu) if torch.cuda.is_available() else torch.device("cpu")  # 0, 1
    # Check that it took effect:
    print("Current device?", torch.cuda.current_device())
    print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))
    ############################################

    bitesAndBytes_config = {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16
    }
    ############################################
    ROOT_PATH = "../../CBMS_MentalHealth_FeatureSelection"
    access_token = "..." # Your HuggingFace Access token
    # Load dataset:
    DS = "counseilChat" # counseilChat 7cups
    splitName = "complete_all_TopicClassification.csv"  # "test/eval/train.csv"

    MODEL_PATH = 'klyang/MentaLLaMA-chat-7B'  # 'meta-llama/Llama-2-7b-chat-hf'
    max_num_tokens = 2048 - 1024 - 218 - 100  # max number of tokens - expected answer [it repeats question, hence -1024] - prompt size - novel content (75 words - 100 tokens)

    EXTRA_NAME = ""  #
    path_dataset = os.path.join(ROOT_PATH, "data/" + DS + "/dataSplits/" + splitName)

    column2extractText = "text"
    # save path:
    root_save_path = os.path.join(ROOT_PATH, "data/experiments/embsClusterPooling",MODEL_PATH.replace("/", "_"), DS, EXTRA_NAME)
    os.makedirs(root_save_path, exist_ok=True)
    out_path_predictions = os.path.join(ROOT_PATH, root_save_path, splitName)

    df_test = pd.read_csv(path_dataset, sep=";", header=0)
    dict_topics = get_dicts_DS(DS)


    prompt_template = '"<DATA2INSERT>"'


    outputs2save = ["last_hidden_state", "pooler_output"]
    bnb_config = BitsAndBytesConfig(
        **bitesAndBytes_config)

    # Load model
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, truncation_side="left", token=access_token)
    model = LlamaModel.from_pretrained(MODEL_PATH, quantization_config=bnb_config, token=access_token, device_map="auto") #device_map="auto",
    #model.to(device)
    model.eval()
    print(model)

    ## SEE/CHECK LAYERS OF THE MODEL:
    # Extract predictions:
    with torch.no_grad():
        generate_embs_Llama2(df_test, model, tokenizer, device, max_num_tokens=max_num_tokens,
                                 column2extractText=column2extractText,
                                 root_save_path=root_save_path, prompt_template=prompt_template)







