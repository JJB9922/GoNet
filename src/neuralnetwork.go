/*
Basic neural network with input layer, hidden layer and output layer.
Flexible and extensible.

Will be classifying Iris flowers using Fisher's Iris dataset.
*/

package main

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
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

// Neural network config and matrices for layers.
type neuralNetwork struct {
	config        neuralNetworkConfig
	weightsHidden *mat.Dense
	biasesHidden  *mat.Dense
	weightsOut    *mat.Dense
	biasesOut     *mat.Dense
}

// Initialize the network
func initializeNewNeuralNetwork(config neuralNetworkConfig) *neuralNetwork {
	return &neuralNetwork{config: config}
}

/*
Implementation of a sigmoid function.

This function squishes down any number to be between 0 and 1.

Then we can use that to decide activation (whether a 'neuron' is on or off).
*/
func sigmoidFunction(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// And the derivative - which describes how the output changes wrt input changes.
// This is applied for backpropagation.
func sigmoidPrimeDerivative(x float64) float64 {
	return sigmoidFunction(x) * (1.0 - sigmoidFunction(x))
}

/*
Implementation of the training.
1. Initialize weights / biases.
2. Forward-feed training data.
3. Compare output to correct output to determine error.
4. Calcualte changes to weights / biases based on above.
5. Backpropagate changes through the network.
6. Repeat 2-5 for X epochs or until a termination case is reached.

Steps 3-5 will use a Stochastic Gradient Descent.
This takes a random batch of the data, then uses a gradient descent
algorithm to get the direction the parameters need to move.
*/
func (nn *neuralNetwork) train(x, y *mat.Dense) error {
	//Initialise weights / biases.
	randomSource := rand.NewSource(time.Now().UnixNano())
	randomGenerator := rand.New(randomSource)

	weightsHidden := mat.NewDense(nn.config.inputNeurons,
		nn.config.hiddenNeurons,
		nil)
	biasesHidden := mat.NewDense(1,
		nn.config.hiddenNeurons,
		nil)

	weightsOut := mat.NewDense(nn.config.hiddenNeurons,
		nn.config.outputNeurons,
		nil)
	biasesOut := mat.NewDense(1,
		nn.config.outputNeurons,
		nil)

	weightsHiddenRaw := weightsHidden.RawMatrix().Data
	biasesHiddenRaw := biasesHidden.RawMatrix().Data

	weightsOutRaw := weightsOut.RawMatrix().Data
	biasesOutRaw := biasesOut.RawMatrix().Data

	for _, param := range [][]float64{
		weightsHiddenRaw,
		biasesHiddenRaw,
		weightsOutRaw,
		biasesOutRaw,
	} {
		for i := range param {
			param[i] = randomGenerator.Float64()
		}
	}

	// Output of the neural network.
	output := new(mat.Dense)

	// Implement backpropagation.
	if err := nn.backpropagate(x, y, weightsHidden, biasesHidden,
		weightsOut, biasesOut, output); err != nil {
		return err
	}

	// Define trained network
	nn.weightsHidden = weightsHidden
	nn.biasesHidden = biasesHidden
	nn.weightsOut = weightsOut
	nn.biasesOut = biasesOut

	return nil
}

// Backpropagation - could be optimized by creating less matrices.
func (nn *neuralNetwork) backpropagate(x, y, weightsHidden,
	biasesHidden, weightsOut, biasesOut, output *mat.Dense) error {

	// Loop over X epochs
	for i := 0; i < nn.config.numEpochs; i++ {

		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, weightsHidden)
		addBiasesHidden := func(_, col int, v float64) float64 {
			return v + biasesHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBiasesHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 {
			return sigmoidFunction(v)
		}
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, weightsOut)
		addBiasesOut := func(_, col int, v float64) float64 {
			return v + biasesOut.At(0, col)
		}
		outputLayerInput.Apply(addBiasesOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Close the loop
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 {
			return sigmoidPrimeDerivative(v)
		}
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, weightsOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters
		weightsOutAdjusted := new(mat.Dense)
		weightsOutAdjusted.Mul(hiddenLayerActivations.T(), dOutput)
		weightsOutAdjusted.Scale(nn.config.learningRate, weightsOutAdjusted)
		weightsOut.Add(weightsOut, weightsOutAdjusted)

		biasesOutAdjusted, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		biasesOutAdjusted.Scale(nn.config.learningRate, biasesOutAdjusted)
		biasesOut.Add(biasesOut, biasesOutAdjusted)

		weightsHiddenAdjusted := new(mat.Dense)
		weightsHiddenAdjusted.Mul(x.T(), dHiddenLayer)
		weightsHiddenAdjusted.Scale(nn.config.learningRate, weightsHiddenAdjusted)
		weightsHidden.Add(weightsHidden, weightsHiddenAdjusted)

		biasesHiddenAdjusted, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		biasesHiddenAdjusted.Scale(nn.config.learningRate, biasesHiddenAdjusted)
		biasesHidden.Add(biasesHidden, biasesHiddenAdjusted)
	}

	return nil
}

// Helper to sum a matrix along one dimension
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(1, numCols, data)
	default:
		return nil,
			errors.New("invalid axis - must be 0 or 1")
	}

	return output, nil
}

// Make a prediction using the trained netowrk
func (nn *neuralNetwork) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check that a trained model is being used
	if nn.weightsHidden == nil || nn.weightsOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}

	if nn.biasesHidden == nil || nn.biasesOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	output := new(mat.Dense)

	// Feed forward
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.weightsHidden)
	addBiasesHidden := func(_, col int, v float64) float64 {
		return v + nn.biasesHidden.At(0, col)
	}
	hiddenLayerInput.Apply(addBiasesHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 {
		return sigmoidFunction(v)
	}
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.weightsOut)
	addBiasesOut := func(_, col int, v float64) float64 {
		return v + nn.biasesOut.At(0, col)
	}
	outputLayerInput.Apply(addBiasesOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}
