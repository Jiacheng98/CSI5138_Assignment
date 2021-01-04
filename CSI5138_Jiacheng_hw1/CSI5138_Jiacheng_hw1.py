# Name: Jiacheng Hou
# Student Number: 300125708

import math
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import itertools


def getData(N, variance):
	dtype = torch.float
	x = torch.rand(N, 1, dtype=dtype)
	sd = math.sqrt(variance)
	noise = torch.empty(N, 1,  dtype=dtype).normal_(mean=0.0,std=sd)
	y = torch.cos(2*math.pi*x) + noise
	return x,y


def getBatch(x, y, dataset_indices, batch_size, batch_number):
	beginning_index = batch_number * batch_size
	ending_index = beginning_index + batch_size
	if ending_index > len(dataset_indices):
		return x[dataset_indices[beginning_index:]], y[dataset_indices[beginning_index:]]
	return x[dataset_indices[beginning_index:ending_index]], y[dataset_indices[beginning_index:ending_index]]

def getMSE(input, target):
	return torch.mean((input - target).pow(2))

def make_features(x, samples, poly_degree):
    x = x.unsqueeze(1)
    return torch.reshape(torch.cat([x ** i for i in range(1, poly_degree+1)], 1), (samples, poly_degree))


def fitData(x_training, y_training, degree, variance, training_size, batch_size):
	fc = torch.nn.Linear(x_training[0].size(0), 1)
	learning_rate = 1e-4
	epoch = 50

	training_dataset_size = len(x_training)
	training_dataset_indices = list(range(training_dataset_size))

	E_in = 0.0

	for e in range(epoch):
		batches = (int) (math.ceil(training_size / batch_size))

		np.random.shuffle(training_dataset_indices)

		for batch in range(batches):
			x_batch, y_batch = getBatch(x_training, y_training, training_dataset_indices, batch_size, batch)
			reg = 0
			reg_lambda = 1e-1
			output_test = getMSE(fc(x_batch), y_batch) 
			for parameter in fc.parameters():
				reg += 0.5*(parameter ** 2).sum()
			output = getMSE(fc(x_batch), y_batch) + reg_lambda*reg
			loss = output.item()
			fc.zero_grad()
			output.backward()
			with torch.no_grad():
				for param in fc.parameters():
					param -= learning_rate * param.grad


			E_in += loss

	E_in /= (epoch*batches)

	testing_size = 1000
	x_testing_origin, y_testing = getData(testing_size,variance)
	if degree != 0:
		x_testing = make_features(x_testing_origin, testing_size, degree)
	else:
		x_testing = x_testing_origin

	testing_prediction = fc(x_testing)
	output_test = getMSE(testing_prediction, y_testing)
	E_out = output_test.item()

	dictVar = dict()
	dictVar['Bias'] = fc.bias[0].item()

	for r in range(degree):
		dictVar['Weight' + str(r+1)] = fc.weight[0][r].item()
	dictVar['E_in'] = E_in
	dictVar['E_out'] = E_out

	return dictVar



def experiment(N, d, variance, batch_size):
	M = 50
	E_total_in, E_total_out, bias_total= 0.0, 0.0, 0.0
	weights_dict = dict()

	for weight_number1 in range(d):
		weights_dict['Weight' + str(weight_number1+1)] = 0.0

	for trial in range(M):
		trials_training_dataset_x, trials_training_dataset_y = getData(N, variance)
		if d != 0:
			trials_training_dataset_x = make_features(trials_training_dataset_x, N, d)
		result = fitData(trials_training_dataset_x, trials_training_dataset_y, d, variance, N, batch_size)
		E_total_in += result['E_in']
		E_total_out += result['E_out']
		bias_total += result['Bias']

		for each_weight in range(d):
			weights_dict['Weight' + str(each_weight + 1)] += result['Weight' + str(each_weight + 1)]

	E_in_average = E_total_in/M
	E_out_average = E_total_out/M
	bias_average = bias_total/M

	for weight_number2 in range(d):
		weights_dict['Weight' + str(weight_number2+1)] /= M

	another_testing_dataset_x_origin, another_testing_dataset_y = getData(1000, variance)
	if d != 0:
		another_testing_dataset_x = make_features(another_testing_dataset_x_origin, 1000, d)
	else:
		another_testing_dataset_x = another_testing_dataset_x_origin

	model = torch.nn.Linear(another_testing_dataset_x[0].size(0), 1)

	with torch.no_grad():
		for count in range(d):
			model.weight[0, count] = weights_dict['Weight' + str(count+1)]
		model.bias[0] = bias_average

	another_testing_prediction = model(another_testing_dataset_x)
	output_another_training = getMSE(another_testing_prediction, another_testing_dataset_y)
	E_bias = output_another_training.item()

	return E_in_average, E_out_average, E_bias



def plotFigure(N_list, d_list, variance_list, x_label, title, x_list):
	E_in_list, E_out_list, E_bias_list = [], [], []
	combination_list = list(itertools.product(N_list, d_list, variance_list))
	for combination in combination_list:
		size = combination[0]
		poly = combination[1]
		vari = combination[2]
		experiment_result = experiment(size, poly, vari, 10)
		E_in_list.append(experiment_result[0])
		E_out_list.append(experiment_result[1])
		E_bias_list.append(experiment_result[2])
		print(" Size of training dataset: " + str(size) + "\n Degree of training dataset: " + str(poly) + \
		"\n Variance of training dataset: " + str(vari) + "\n E_in, E_out, E_bias" + str(experiment_result) + "\n\n")

	plt.figure()
	plt.plot(x_list, E_in_list, label = 'Average E_in')
	plt.plot(x_list, E_out_list, color = "red", label = 'Average E_out')
	plt.plot(x_list, E_bias_list, color = "black", label = 'E_bias')
	if(x_label == "Training Dataset Size"):
		plt.xticks(np.arange(0, 220, 20), fontsize=6)
	elif (x_label == "Polynomial Degree"):
		plt.xticks(np.arange(min(x_list), max(x_list)+1, 1), fontsize = 6)
	plt.yticks(np.arange(0, 1.5, step=0.1)) 
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel('Mean Square Error')
	plt.legend()
	plt.savefig(x_label + "_weight_decay")
	plt.close()

	return E_in_list, E_out_list, E_bias_list


N_list1 = [5]
d_list1 = list(range(31))
variance_list1 = [0.0001]
x_label1 = "Polynomial Degree"
title1 = "Training Dataset Size= " + str(N_list1[0]) + ", Variance = " + str(variance_list1[0])
plotFigure(N_list1, d_list1, variance_list1, x_label1, title1, d_list1)

N_list2 = [2, 5, 10, 20, 50, 100, 200]
d_list2 = [8]
variance_list2 = [0.0001]
x_label2 = "Training Dataset Size"
title2 = "Polynomial Degree = " + str(d_list2[0]) + ", Variance = " + str(variance_list2[0])
plotFigure(N_list2, d_list2, variance_list2, x_label2, title2, N_list2)


N_list3 = [200]
d_list3 = [8]
variance_list3 = [0.0001, 0.01, 1]
x_label3 = "Noise Variance"
title3 = "Training Dataset Size = " + str(N_list3[0]) + ", Polynomial Degree = " + str(d_list3[0])
plotFigure(N_list3, d_list3, variance_list3, x_label3, title3, variance_list3)