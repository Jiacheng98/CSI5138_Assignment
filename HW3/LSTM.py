import argparse
import os

import torch
from torchtext import data, datasets
import torch.nn as nn
import torch.optim as optim

import random
import matplotlib.pyplot as plt
import numpy as np

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        # embedded = [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(text))

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hidden dimenstion]
        # cell = [num layers * num directions, batch size, hidden dimenstion]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

	    # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
	    # and apply dropout
        hidden = self.dropout(
	        torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))


        
        final_output = self.linear(hidden)
        return final_output


def main():

    TEXT = data.Field(tokenize='spacy', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    # print(vars(train_data.examples[0]))

    train_data, valid_data = train_data.split(random_state = random.seed(SEED))

    # print(f'Number of training examples: {len(train_data)}')
    # print(f'Number of validation examples: {len(valid_data)}')
    # print(f'Number of testing examples: {len(test_data)}')

    MAX_VOCAB_SIZE = 25_000
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.50d",
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    # print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    # print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    BATCH_SIZE = 64

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        device=device)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 50
    HIDDEN_DIM = 500
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.4
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
                BIDIRECTIONAL, DROPOUT, PAD_IDX)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    pretrained_embeddings = TEXT.vocab.vectors
    print(f'The shape of pretrained_embeddings: {pretrained_embeddings.shape}')
    # replace the initial weights of the embedding layer with the pre-trained embeddings
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # setting <unk>, <pad> tokens in the embedding weights matrix to zeros
    # The index of the <pad> token passed to the padding_idx of the embedding layer, it will remain zeros throughout training
    # The <unk> token embedding will be learned.
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 10
    best_valid_loss = float('inf')



    train_accuracys = []
    validation_accuracys = []
    train_losses = []
    validation_losses = []
    epoch_list = [i for i in range(1, N_EPOCHS+1)]

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        # test_loss, test_acc = evaluate(model, test_iterator, criterion)

        train_accuracys.append(train_acc)
        train_losses.append(train_loss)
        validation_accuracys.append(valid_acc)
        validation_losses.append(valid_loss)



        if valid_loss < best_valid_loss:
	        best_valid_loss = valid_loss
	        torch.save(model.state_dict(), 'LSTM_500.pt')

        print(f'Epoch {epoch+1}:')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Validation Loss: {valid_loss:.3f} |  Validation Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('LSTM_500.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


    fig = plt.figure()
    plt.plot(epoch_list, train_losses, color='black')
    plt.plot(epoch_list, validation_losses, color='pink')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Binary cross-entropy loss')
    plt.yticks(np.arange(0, 1.1, step=0.1)) 
    plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1)) 
    plt.title('Hidden Dimension 500')
    plt.savefig('LSTM_loss_500')
    plt.close()

    fig2 = plt.figure()
    plt.plot(epoch_list, train_accuracys, color='black')
    plt.plot(epoch_list, validation_accuracys, color='pink')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right', bbox_to_anchor=(1.1, 1.15))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1))  
    plt.title('Hidden Dimension 500')
    plt.savefig('LSTM_acc_500')
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    # convert into float for division
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss/len(iterator), epoch_acc/len(iterator)


if __name__ == "__main__":

    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
