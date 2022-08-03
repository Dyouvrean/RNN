import torch.nn as nn
import torch.nn.functional as F
import torch


class DOC_RNN(nn.Module):

    def __init__(self, embedding_dim, lstm_hidden_dim, number_of_labels):
        # Initialize all of the layers the forward method will use:
        super().__init__()

        # Input already has tokens mapped to vectors: no embedding layer needed.
        # For getting LSTM hidden states from embedding vectors:
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, bidirectional=True,batch_first=True)
        # Transforms the embedding to a vector of length number_of_tags
        self.linearClassifier = nn.Linear(lstm_hidden_dim * 4, 256)
        self.fc2 = nn.Linear(256, number_of_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        # Defines how the LSTM will run:
        # X: a list of FloatTensors: [tensor([...]), tensor([...]), ...]
        #   each tensor contains a matrix of input embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        doc_vecs = []  # will hold the final hidden state of each document of X.
        for doc in X:  # doc is a FloatTensor of all input embeddings for a single sequence
            doc = doc.clone().detach().cuda()
            doc.to(device)
            # print(doc.shape)
            # print(doc.unsqueeze(1).shape)
            s, _ = self.lstm(doc.unsqueeze(1))
            # s is now the outputs for all words in the doc
            # print(s.shape)
            s = s.squeeze(1)
            avg_pool = torch.mean(s, 0,True)
            #avg_pool = torch.mean(avg_pool, 0, True)
            #print(avg_pool.shape)
            # print(avg_pool.dtype)
            max_pool, _ = torch.max(s, 0,True)
            #max_pool, _ = torch.max(max_pool, 0, True)
            # print(max_pool.shape)
            # print(max_pool.dtype)
            # temp= torch.cat((s[-2, :, :], s[-1, :, :]),1)
            # print(s[-1].shape)
            # print(temp.shape)
            # print(temp.dtype)
            # doc_vecs.append(s[-1])
            doc_vecs.append(torch.cat((avg_pool, max_pool), 1))  # represent the doc as the the final word output
        # turn the list of document vectors into a matrix (tensor) of size: num_docs x lstm_hidden_dim
        doc_vecs = torch.stack(doc_vecs).squeeze(1)
        # obtain scores for each doc
        # doc_vecs = self.linearClassifier(doc_vecs)
        rel = self.relu(doc_vecs)
        dense1 = self.linearClassifier(rel)
        drop = self.dropout(dense1)
        yprobs = self.fc2(drop)
        # change the scores to probabilities1
        # yprobs = F.softmax(preds, 1)
        return yprobs
