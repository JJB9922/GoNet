package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {

	// File Handler
	f, err := os.Open("../data/training_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// Form matrices
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	// Init architecture and params
	config := neuralNetworkConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Training
	network := initializeNewNeuralNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Parsing test data
	// File Handler
	tf, err := os.Open("../data/test_data.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer tf.Close()

	treader := csv.NewReader(tf)
	reader.FieldsPerRecord = 7

	trawCSVData, err := treader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	testInputsData := make([]float64, 4*len(trawCSVData))
	testLabelsData := make([]float64, 3*len(trawCSVData))

	var tinputsIndex int
	var tlabelsIndex int

	for idx, record := range trawCSVData {
		if idx == 0 {
			continue
		}
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i == 4 || i == 5 || i == 6 {
				testLabelsData[tlabelsIndex] = parsedVal
				tlabelsIndex++
				continue
			}

			testInputsData[tinputsIndex] = parsedVal
			tinputsIndex++
		}
	}

	testInputs := mat.NewDense(len(trawCSVData), 4, testInputsData)
	testLabels := mat.NewDense(len(trawCSVData), 3, testLabelsData)

	// Make predictions using trained model
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate model accuracy
	var truePosNeg int
	numPredictions, _ := predictions.Dims()
	for i := 0; i < numPredictions; i++ {
		labelRow := mat.Row(nil, i, testLabels)
		var species int
		for idx, label := range labelRow {
			if label == 1.0 {
				species = idx
				break
			}
		}

		// Accumulate count
		if predictions.At(i, species) ==
			floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	accuracy := float64(truePosNeg) / float64(numPredictions)
	fmt.Printf("\nAccuracy %0.2f\n\n", accuracy*100)

	// Print weights/biases for fun
	fmt.Print("Final Weights & Biases:\n")
	for i := 0; i < len(network.weightsOut.RawMatrix().Data); i++ {
		fmt.Print("W: ", network.weightsOut.RawMatrix().Data[i])
		fmt.Print("\n")
	}
	fmt.Print("\n")
	for i := 0; i < len(network.biasesOut.RawMatrix().Data); i++ {
		fmt.Print("B: ", network.biasesOut.RawMatrix().Data[i])
		fmt.Print("\n")
	}

}
