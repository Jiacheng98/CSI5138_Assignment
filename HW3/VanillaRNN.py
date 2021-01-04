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
	def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
		super().__init__()
		self.embedding = nn.Embedding(input_dim, embedding_dim)
		self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                           bidirectional=bidirectional, dropout=dropout)
		self.linear = nn.Linear(hidden_dim * 2, output_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, text):
		#text = [sent len, batch size]
		#embedded = [sent len, batch size, emb dim]
		embedded = self.dropout(self.embedding(text))

		# output = [sent len, batch size, hid dim]
		# hidden = [1, batch size, hid dim]
		output, hidden = self.rnn(embedded)
		# print(f'The shape of hidden: {hidden.shape}') #[2, 64, 20], if add bidirectional, the shape is [6, 64, 20]
		# print(f'Hidden[-1, :, :]: {hidden[-1,:,:]}') # The last hidden state
		# print(f'Output[-1, :, :]: {output[-1,:,:]}') # Same with the previous one

		# For one layer
		# assert torch.equal(output[-1,:,:], hidden.squeeze(0)) 
		# hidden_output = hidden.squeeze(0) 

		#For >= two layers
		# hidden_output = self.dropout(hidden[-1, :, :])

		# For bidirectional
		hidden_output = self.dropout(
	        torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)) 

		final_output = self.linear(hidden_output)
		return final_output



def main():

	TEXT = data.Field(tokenize = 'spacy')
	LABEL = data.LabelField(dtype = torch.float)

	train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
	# print(vars(train_data.examples[0]))

	train_data, valid_data = train_data.split(random_state = random.seed(SEED))

	# print(f'Number of training examples: {len(train_data)}')
	# print(f'Number of validation examples: {len(valid_data)}')
	# print(f'Number of testing examples: {len(test_data)}')
	# print("Train data example: " + str(len(train_data.examples)))


	MAX_VOCAB_SIZE = 25_000
	TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)
	LABEL.build_vocab(train_data)
	# print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
	# print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")


	BATCH_SIZE = 64

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
	    (train_data, valid_data, test_data), 
	    batch_size = BATCH_SIZE,
	    sort_within_batch=True,
	    device = device)



	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 100
	HIDDEN_DIM = 500
	OUTPUT_DIM = 1
	N_LAYERS = 2
	BIDIRECTIONAL = True
	DROPOUT = 0.25

	model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
                BIDIRECTIONAL, DROPOUT)
	print(f'The model has {count_parameters(model):,} trainable parameters')

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
		train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
		valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
		# test_loss, test_acc = evaluate(model, test_iterator, criterion)

		train_accuracys.append(train_acc)
		train_losses.append(train_loss)
		validation_accuracys.append(valid_acc)
		validation_losses.append(valid_loss)

		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), 'Vanilla2_500.pt')


		print(f'Epoch {epoch+1}:')
		print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'\t Validation Loss: {valid_loss:.3f} |  Validation Acc: {valid_acc*100:.2f}%')


	model.load_state_dict(torch.load('Vanilla2_500.pt'))
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
	plt.savefig('VanillaRNN_loss_500')
	plt.close()

	fig2 = plt.figure()
	plt.plot(epoch_list, train_accuracys, color='black')
	plt.plot(epoch_list, validation_accuracys, color='pink')
	plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper right')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.yticks(np.arange(0, 1.1, step=0.1))
	plt.xticks(range(min(epoch_list), max(epoch_list)+1, 1))  
	plt.title('Hidden Dimension 500')
	plt.savefig('VanillaRNN_acc_500')
	plt.close()






def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    #convert into float for division
    correct = (rounded_preds == y).float()  
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
	epoch_loss = 0
	epoch_acc = 0

	model.train()

	for batch in iterator:
		optimizer.zero_grad()
		predictions = model(batch.text).squeeze(1)
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
			predictions = model(batch.text).squeeze(1)
			loss = criterion(predictions, batch.label)
			acc = binary_accuracy(predictions, batch.label)

			epoch_loss += loss.item()
			epoch_acc += acc.item()

	return epoch_loss/len(iterator), epoch_acc/len(iterator)




if __name__=="__main__":
      
    
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main()
