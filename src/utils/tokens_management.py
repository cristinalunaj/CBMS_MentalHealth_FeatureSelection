import re
from nltk.corpus import stopwords

def nTokens_is_lowerThanMaximum(sentence, tokenizer, max_num_tokens = 512):
    tokens_sentence = tokenizer(sentence, return_tensors="pt")
    tokens_sentence = tokens_sentence.data["input_ids"].detach()
    n_tokens = tokens_sentence.shape[-1]
    if(n_tokens>max_num_tokens):
        # remove new word:
        print("TOKENS:", str(n_tokens))
        del tokens_sentence
        return False
    else:
        del tokens_sentence
        return True

def remove_encoding_word(word):
    word = str(word)
    word = word.encode('ASCII', 'ignore').decode('ASCII')
    return word



def remove_encoding_text(text, exta_words):
    text = str(text)
    stop = stopwords.words('english') + exta_words + stopwords.words('spanish')
    text = ' '.join(remove_encoding_word(word) for word in text.split() if word not in stop)
    return text


def remove_words(prompt, tokenizer, max_num_tokens = 512):
    while(not nTokens_is_lowerThanMaximum(prompt, tokenizer, max_num_tokens = max_num_tokens)):
        # remove first word from initial prompt
        prompt = prompt.split(" ", 1)[-1]
    return prompt

def charactersChecker(splitted_prompt):
    final_words_in_prompt = []
    for word in splitted_prompt:
        word = word.strip()
        if(len(word)<=45):
            #append:
            final_words_in_prompt.append(word)
        elif(len(word)>45 and 'http' in word): # maintain links
            final_words_in_prompt.append(word)
        else:
            print("REMOVED WORD: ", word)
    return final_words_in_prompt




def cleanAndSplitSentence(prompt):
    # longest word in english has 45 chracters
    prompt = cleanSentence(prompt)
    print(prompt)# remove text between HTML symbols of comments
    splitted_prompt = prompt.split(" ")
    splitted_prompt = charactersChecker(splitted_prompt)
    return splitted_prompt, prompt


def cleanSentence(prompt):
    # longest word in english has 45 chracters
    prompt = str(prompt)
    prompt = prompt.split("\n\n\n")[-1]
    # Clean prompt
    prompt = prompt.replace("----------------", "")
    prompt = prompt.replace("\n", " ")
    prompt = re.sub("<!--.*?-->", '', prompt)
    return prompt


def split_sentence(prompt, tokenizer, max_num_tokens, words_in_window = 300,sliding_window=0.5):
    #approx_words_in_window = 512 * 3/4 #384 words
    initial_idx = 0

    splitted_prompt,_ = cleanAndSplitSentence(prompt)
    list_sub_prompts = []
    step = int(words_in_window*sliding_window)
    end_idx = min(len(splitted_prompt), initial_idx + words_in_window)
    if(words_in_window>len(splitted_prompt)):
        list_sub_prompts.append(" ".join(splitted_prompt[initial_idx:end_idx]))
    else:
        while (end_idx < len(splitted_prompt)):
            # remove first word from initial prompt
            prompt = " ".join(splitted_prompt[initial_idx:end_idx])
            initial_idx += step
            end_idx = min(len(splitted_prompt), end_idx+step)
            # check that the new prompt is okay in size & add to the list of sub-prompts:
            if(nTokens_is_lowerThanMaximum(prompt, tokenizer, max_num_tokens=max_num_tokens)):
                list_sub_prompts.append(prompt)
            else:
                print("MUY LARGO!!!")
                print(prompt)
                list_sub_prompts.append(prompt)
    return list_sub_prompts
