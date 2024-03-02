/*
Basic neural network with input layer, hidden layer and output layer.
Flexible and extensible.

Will be classifying Iris flowers using Fisher's Iris dataset.
*/

package main

import (
	"gonum.org/v1/gonum/mat"
)

// Architecture of neural network with learning parameters.
type neuralNetworkConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

// Neural netowkr config and matrices for layers.
type neuralNetwork struct {
	config  neuralNetworkConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

func initializeNewNeuralNetwork(config neuralNetworkConfig) *neuralNetwork {
	return &neuralNetwork{config: config}
}

func main() {

}
