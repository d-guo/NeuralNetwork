# 2 layer neural network with 5 neurons in hidden layer
#approximate f(x) = 2*relu(-x + 1) - 1

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

#lists containing the data to be plotted for visualizing the convergence of the averageloss
xpoints = list()
ypoints = list()
plt.ion()

def func(x):
	return 2 * max(0, -x + 1) - 1

class neuralnet:
	def __init__(self):	#input layer has 1 neuron, hidden layer has 5 neurons, output layer has 1 neuron
		self.weights = list()	#weights will be 2d list with rows as layer and columns as neuron
		self.biases = list()	#biases will be list with 1 bias per layer
		self.nodevals = list()	#3d list labels for 2d lists with rows as layer and columns as neuron

		self.weights.append(list())
		self.weights.append(list())

		#give random values for weights and biases
		for i in range(0, 2):
			self.biases.append(random.uniform(0, 2))
			for j in range(0, 5):
				self.weights[i].append(random.uniform(0, 2))

	def averageloss(self, desiredOutputs):
		totalLoss = 0
		numofits = len(self.nodevals)

		for i in range(0, numofits):
			totalLoss += (self.nodevals[i][2][0] - desiredOutputs[i]) ** 2

		return totalLoss / numofits

	def relu(self, x):
		return max(0, x)

	def drelu(self, x):
		if(x > 0):
			return 1
		else:
			return 0

	def feedforward(self, inputs):	#updates nodevals with new inputs
		self.nodevals.clear()
		numofits = len(inputs)

		for i in range(0, numofits):
			#ith input label
			self.nodevals.append(list())

			#3 empty lists for input layer, hidden layer, output layer values
			self.nodevals[i].append(list())
			self.nodevals[i].append(list())
			self.nodevals[i].append(list())

			#adding values
			self.nodevals[i][0].append(inputs[i])
			self.nodevals[i][1].append(self.relu(self.weights[0][0] * self.nodevals[i][0][0] + self.biases[0]))
			self.nodevals[i][1].append(self.relu(self.weights[0][1] * self.nodevals[i][0][0] + self.biases[0]))
			self.nodevals[i][1].append(self.relu(self.weights[0][2] * self.nodevals[i][0][0] + self.biases[0]))
			self.nodevals[i][1].append(self.relu(self.weights[0][3] * self.nodevals[i][0][0] + self.biases[0]))
			self.nodevals[i][1].append(self.relu(self.weights[0][4] * self.nodevals[i][0][0] + self.biases[0]))
			self.nodevals[i][2].append(self.weights[1][0] * self.nodevals[i][1][0] + self.weights[1][1] * self.nodevals[i][1][1] + self.weights[1][2] * self.nodevals[i][1][2] + self.weights[1][3] * self.nodevals[i][1][3] + self.weights[1][4] * self.nodevals[i][1][4] + self.biases[1])

		return self.nodevals

	def dzdw(self):	#change in output per change in weights
		dzdw = list()	#3d list labels for 2d lists with rows as layers and columns as neuron
		numofits = len(self.nodevals)

		for i in range(0, numofits):
			dzdw.append(list())
			dzdw[i].append(list())
			dzdw[i].append(list())

			#first layer
			dzdw[i][0].append(self.weights[1][0] * self.drelu(self.weights[0][0] * self.nodevals[i][0][0] + self.biases[0]) * self.nodevals[i][0][0])
			dzdw[i][0].append(self.weights[1][1] * self.drelu(self.weights[0][1] * self.nodevals[i][0][0] + self.biases[0]) * self.nodevals[i][0][0])
			dzdw[i][0].append(self.weights[1][2] * self.drelu(self.weights[0][2] * self.nodevals[i][0][0] + self.biases[0]) * self.nodevals[i][0][0])
			dzdw[i][0].append(self.weights[1][3] * self.drelu(self.weights[0][3] * self.nodevals[i][0][0] + self.biases[0]) * self.nodevals[i][0][0])
			dzdw[i][0].append(self.weights[1][4] * self.drelu(self.weights[0][4] * self.nodevals[i][0][0] + self.biases[0]) * self.nodevals[i][0][0])

			#second layer
			dzdw[i][1].append(self.nodevals[i][1][0])
			dzdw[i][1].append(self.nodevals[i][1][1])
			dzdw[i][1].append(self.nodevals[i][1][2])
			dzdw[i][1].append(self.nodevals[i][1][3])
			dzdw[i][1].append(self.nodevals[i][1][4])

		return dzdw

	def dzdb(self): #change in output per change in biases
		dzdb = list()
		numofits = len(self.nodevals)

		for i in range(0, numofits):
			dzdb.append(list())
			dzdb[i].append(self.weights[1][0] * self.drelu(self.weights[0][0] * self.nodevals[i][0][0] + self.biases[0]) + self.weights[1][1] * self.drelu(self.weights[0][1] * self.nodevals[i][0][0] + self.biases[0]) + self.weights[1][2] * self.drelu(self.weights[0][2] * self.nodevals[i][0][0] + self.biases[0]) + self.weights[1][3] * self.drelu(self.weights[0][3] * self.nodevals[i][0][0] + self.biases[0]) + self.weights[1][4] * self.drelu(self.weights[0][4] * self.nodevals[i][0][0] + self.biases[0]))
			dzdb[i].append(1)

		return dzdb

	def backprop(self, desiredOutputs, learningrate):
		numofits = len(self.nodevals)

		#find change in cost per change in weights
		dzdw = self.dzdw()
		dcdw = list()
		for i in range(0, numofits):
			dcdz = 2 * (self.nodevals[i][2][0] - desiredOutputs[i])
			dcdw.append(list())

			dcdw[i].append(list())
			dcdw[i].append(list())

			dcdw[i][0].append(dcdz * dzdw[i][0][0])
			dcdw[i][0].append(dcdz * dzdw[i][0][1])
			dcdw[i][0].append(dcdz * dzdw[i][0][2])
			dcdw[i][0].append(dcdz * dzdw[i][0][3])
			dcdw[i][0].append(dcdz * dzdw[i][0][4])

			dcdw[i][1].append(dcdz * dzdw[i][1][0])
			dcdw[i][1].append(dcdz * dzdw[i][1][1])
			dcdw[i][1].append(dcdz * dzdw[i][1][2])
			dcdw[i][1].append(dcdz * dzdw[i][1][3])
			dcdw[i][1].append(dcdz * dzdw[i][1][4])

		#find change in cost per change in biases
		dzdb = self.dzdb()
		dcdb = list()
		for i in range(0, numofits):
			dcdz = 2* (self.nodevals[i][2][0] - desiredOutputs[i])
			dcdb.append(list())

			dcdb[i].append(dcdz * dzdb[i][0])
			dcdb[i].append(dcdz * dzdb[i][1])

		#find average of dcdw
		dcdwavg = list()
		dcdwavg.append(list())
		dcdwavg.append(list())
		dcdwtotal = 0
		for k in range(0, 5):
			for j in range(0, 2):
				for i in range(0, numofits):
					dcdwtotal += dcdw[i][j][k]
				dcdwavg[j].append(dcdwtotal / numofits)
				dcdwtotal = 0

		#find average of dcdb
		dcdbavg = list()
		dcdbtotal = 0
		for j in range(0, 2):
			for i in range(0, numofits):
				dcdbtotal += dcdb[i][j]
			dcdbavg.append(dcdbtotal / numofits)
			dcdbtotal = 0

		#update weights
		for i in range(0, 2):
			for j in range(0, 5):
				self.weights[i][j] -= learningrate * dcdwavg[i][j]

		#update biases
		for i in range(0, 2):
			self.biases[i] -= learningrate * dcdbavg[i]
	
	def train1(self, inputs, desiredOutputs, learningrate):
		self.feedforward(inputs)
		self.backprop(desiredOutputs, learningrate)

	def train(self, learningrate, acceptableerror, maxiterations, datasetsize):
		for i in range(0, maxiterations):
			#generate dataset for training
			dataset = list()
			for j in range(0, datasetsize):
				dataset.append(random.uniform(-100, 100))
			#generate solutionset for training
			solutionset = list()
			for j in range(0, datasetsize):
				solutionset.append(func(dataset[j]))
			#training
			self.train1(dataset, solutionset, learningrate)

			#graph to visualize the averageloss convergence
			if(i % 100 == 0):
				xpoints.append(i)
				ypoints.append(self.averageloss(solutionset))
				plt.gcf().clear()
				plt.plot(xpoints, ypoints)
				plt.draw()
				plt.pause(0.0000001)
				#print(self.averageloss(solutionset))

			#save weights and biases if error is acceptable
			if(self.averageloss(solutionset) < acceptableerror):
				f = open("trainednetworkparameters.txt", "w+")
				for weight in self.weights:
					f.write(str(weight[0]))
					f.write(" ")
					f.write(str(weight[1]))
					f.write(" ")
					f.write(str(weight[2]))
					f.write(" ")
					f.write(str(weight[3]))
					f.write(" ")
					f.write(str(weight[4]))
					f.write("\n")
				f.write(str(self.biases[0]))
				f.write(" ")
				f.write(str(self.biases[1]))
				f.write("\n")
				f.write("averageloss: " + str(self.averageloss(solutionset)))
				f.write("\n")
				f.write("iterations: " + str(i + 1))
				f.close()
				return

			#if training ends at maxiterations, use the final results, even if error is not acceptable
			if(i == maxiterations - 1):
				f = open("trainednetworkparameters.txt", "w+")
				for weight in self.weights:
					f.write(str(weight[0]))
					f.write(" ")
					f.write(str(weight[1]))
					f.write(" ")
					f.write(str(weight[2]))
					f.write(" ")
					f.write(str(weight[3]))
					f.write(" ")
					f.write(str(weight[4]))
					f.write("\n")
				f.write(str(self.biases[0]))
				f.write(" ")
				f.write(str(self.biases[1]))
				f.write("\n")
				f.write("averageloss: " + str(self.averageloss(solutionset)))
				f.write("\n")
				f.write("iterations: " + str(i + 1))
				f.close()
				return

class trainednet:
	def __init__(self):
		self.nn = neuralnet()

		#use saved weights and biases
		f = open("trainednetworkparameters.txt", "r")

		l1 = f.readline()
		l2 = f.readline()
		l3 = f.readline()

		l1 = l1.split()
		l2 = l2.split()
		l3 = l3.split()

		weights1 = list()
		weights2 = list()
		biases = list()

		for i in range(0, 5):
			weights1.append(float(l1[i]))
		for i in range(0, 5):
			weights2.append(float(l2[i]))
		for i in range(0, 2):
			biases.append(float(l3[i]))

		self.nn.weights[0] = weights1
		self.nn.weights[1] = weights2
		self.nn.biases = biases

	def f(self, x):
		return self.nn.weights[1][0] * self.nn.relu(self.nn.weights[0][0] * x + self.nn.biases[0]) + self.nn.weights[1][1] * self.nn.relu(self.nn.weights[0][1] * x + self.nn.biases[0]) + self.nn.weights[1][2] * self.nn.relu(self.nn.weights[0][2] * x + self.nn.biases[0]) + self.nn.weights[1][3] * self.nn.relu(self.nn.weights[0][3] * x + self.nn.biases[0]) + self.nn.weights[1][4] * self.nn.relu(self.nn.weights[0][4] * x + self.nn.biases[0]) + self.nn.biases[1]

#x = -50
#tn = trainednet()
#print(tn.f(x))
#print(func(x))


nn = neuralnet()
nn.train(10e-5, 0.001, 100000, 1000)