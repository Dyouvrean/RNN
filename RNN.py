import pandas as pd
import gensim.downloader as api
import numpy as np
import jieba
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from model import DOC_RNN
from sklearn.metrics import f1_score
from process_stop import remove_stopwords, get_embedding
from load_corpus import get_corpus
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Evaluate import evaluate

# random seed
seed_val = 1
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

dev_set = pd.read_csv("./dev.tsv", sep='\t', names=['text', 'category', 'id'])
train_set = pd.read_csv("./train.tsv", sep='\t', names=['text', 'category', 'id'])
test_set = pd.read_csv("./test.tsv", sep='\t', names=['text', 'category', 'id'])
train_set= train_set.sample(frac = 1)
train_set['text'] = train_set.text.apply(lambda x: " ".join(jieba.cut(x)))
dev_set['text'] = dev_set.text.apply(lambda x: " ".join(jieba.cut(x)))
test_set['text'] = test_set.text.apply(lambda x: " ".join(jieba.cut(x)))

# remove stopwords
stop_words_file = "stopword.txt"
rs_train = remove_stopwords(stop_words_file, train_set)
rs_test = remove_stopwords(stop_words_file, test_set)
rs_dev = remove_stopwords(stop_words_file, dev_set)
general_corpus = get_corpus("sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5")

# model = Word2Vec(sentences=corpus, vector_size=50, window = 5,epochs=50, min_count=5)
# input_dim = len(model.wv) + 1
max_len = 150
max_words = 10000

X_train = get_embedding(rs_train, general_corpus, max_len, len(train_set['text']))
X_test = get_embedding(rs_test, general_corpus, max_len, len(test_set['text']))
X_dev = get_embedding(rs_dev, general_corpus, max_len, len(dev_set['text']))

Y_train = torch.from_numpy(np.array(train_set['category']).astype(np.float32))
Y_test = torch.from_numpy(np.array(test_set['category']).astype(np.float32))
Y_dev = torch.from_numpy(np.array(dev_set['category']).astype(np.float32))


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average="weighted")


batch_size = 32
loader_trainX = DataLoader(X_train,sampler=RandomSampler(X_train),batch_size=batch_size)
loader_trainY = DataLoader(Y_train,sampler=RandomSampler(Y_train),batch_size=batch_size)

loader_testX = DataLoader(X_test,sampler=RandomSampler(X_test),batch_size=batch_size)
loader_testY = DataLoader(Y_test,sampler=RandomSampler(Y_test),batch_size=batch_size)

loader_devX = DataLoader(X_dev,sampler=RandomSampler(X_dev),batch_size=batch_size)
loader_devY = DataLoader(Y_dev,sampler=RandomSampler(Y_dev),batch_size=batch_size)



epochs = 10
print("\nTraining Logistic Regression...")
model = DOC_RNN(300, 256, 5)
sgd_pr = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#X_train = torch.from_numpy(X_train)
X_dev = torch.from_numpy(X_dev)
#X_train.to(device)

# print(device)
# # # training loop:
best_valid_loss = 0
for i in range(epochs):
    model.train()
    sgd_pr.zero_grad()
    loss_train_total = 0

    for X_batch, Y_batch in zip(loader_trainX, loader_trainY):
        model.zero_grad()

        X_batch.to(device)
        Y_batch = Y_batch.cuda()
        outputs = model(X_batch.float())
        loss = loss_func(torch.max(outputs, 1)[0], Y_batch)
        loss_train_total += loss.item()
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sgd_pr.step()



    # forward pass:
    ##ypred_pr = model(X_train.float())
    # ypred_pr_dev = model(X_dev.float())
    #
    # Ytrain = Ytrain.cuda()
    # Ytrain_dev = Ytrain_dev.cuda()
    #
    # loss = loss_func(torch.max(ypred_pr, 1)[0], Ytrain)
    # loss_dev = loss_func(torch.max(ypred_pr_dev, 1)[0], Ytrain_dev)
    # # backward:
    # loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # sgd_pr.step()
    if i % 1 == 0:
        print("  epoch: %d, loss: %.5f" % (i, loss.item()))
        val_loss, predictions, true_vals = evaluate(model,loader_devX,loader_devY,device,loss_func)
        print("val_loss: %.5f" % (val_loss))
        val_f1 = f1_score_func(predictions, true_vals)
        print("f1-score:" + str(val_f1))
        if val_f1 > best_valid_loss:
            best_valid_loss = val_f1
            torch.save(model.state_dict(), f'RNNmodel')

#### evaluate
model.load_state_dict(torch.load('./RNNmodel'))

val_loss, predictions, true_vals = evaluate(model,loader_testX,loader_testY,device,loss_func)
predictions=np.argmax(predictions,axis=1)
print(predictions)
print(f'Accuracy:{len(predictions[predictions == true_vals])}/{len(true_vals)}\n')
print(len(predictions[predictions == true_vals])/len(true_vals))
