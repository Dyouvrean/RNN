import numpy as np


def get_custom_stopwords(stop_words_file):
    with open(stop_words_file, encoding="utf8") as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


def remove_stopwords(stop_words_file, data):
    stopwords = get_custom_stopwords(stop_words_file)
    stopwords.append(' ')
    corpus = []
    for text in data['text']:
        corpus.append([word for word in text if word not in stopwords])
    return corpus


def get_embedding(data, general_corpus, max_len,data_len):
    embedding_matrix = np.zeros((data_len, max_len, 300))
    text_count = 0
    for text in data:
        word_count = 0
        for word in text:
            if word in general_corpus.keys():
                embedding_matrix[text_count][word_count] = general_corpus[word]
            word_count += 1
            if word_count >= max_len:
                break
        text_count += 1
    return embedding_matrix
