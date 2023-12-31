package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/arturobermejo/credeval/dataframe"
	"github.com/arturobermejo/credeval/grad"
)

func parseFloat64(col []any) error {
	for i, val := range col {
		valStr, ok := val.(string)
		if !ok {
			continue
		}

		vFloat, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			continue
		}

		col[i] = vFloat
	}

	return nil
}

func scaleMinMax(col []any) error {
	min := math.Inf(1)
	max := math.Inf(-1)

	for _, val := range col {
		valFloat, ok := val.(float64)
		if !ok {
			continue
		}

		if valFloat < min {
			min = valFloat
		}

		if valFloat > max {
			max = valFloat
		}
	}

	for i, val := range col {
		valFloat, ok := val.(float64)
		if !ok {
			continue
		}

		col[i] = (valFloat - min) / (max - min)
	}

	return nil
}

func main() {
	filename := flag.String("f", "dataset/dataset.csv", "dataset csv file")

	flag.Parse()

	file, err := os.Open(*filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	df, err := dataframe.LoadFromCSV(io.Reader(file))
	if err != nil {
		log.Fatal(err)
	}

	err = df.Transform(0, parseFloat64, scaleMinMax)
	if err != nil {
		log.Fatal(err)
	}

	err = df.Transform(1, parseFloat64)
	if err != nil {
		log.Fatal(err)
	}

	trainSize := int(math.Round(0.8 * float64(len(df.Data[0]))))

	xTrain := df.Data[0][1 : trainSize+1]
	yTrain := df.Data[1][1 : trainSize+1]

	w := grad.NewVar(rand.Float64())
	b := grad.NewVar(rand.Float64())

	params := []*grad.Var{w, b}

	epochs := 50000
	learningRate := 0.01

	for epoch := 1; epoch <= epochs; epoch++ {
		avgGrad := make([]float64, len(params))
		avgLoss := 0.0
		avgAcc := 0.0

		for i := 0; i < trainSize; i++ {
			x := grad.NewVar(xTrain[i].(float64))
			y := grad.NewVar(yTrain[i].(float64))

			xw := grad.Mul(x, w)
			r := grad.Sum(xw, b)

			prob := grad.Sigmoid(r)

			loss, err := grad.BinaryCrossEntropy(prob, y)
			if err != nil {
				log.Fatal(err)
			}

			loss.Backward(1.0)

			for i, param := range params {
				avgGrad[i] += param.Grad() / float64(trainSize)
			}

			avgLoss += loss.Value() / float64(trainSize)

			if math.Round(prob.Value()) == y.Value() {
				avgAcc += 1.0 / float64(trainSize)
			}
		}

		// update params
		for i, param := range params {
			param.SetValue(param.Value() - learningRate*avgGrad[i])
		}

		fmt.Printf("Epoch: %v/%v, loss: %.4f, acc: %.4f\n", epoch, epochs, avgLoss, avgAcc)
	}

	xTest := df.Data[0][trainSize+1:]
	yTest := df.Data[1][trainSize+1:]

	testSize := len(xTest)

	avgAcc := 0.0

	for i := 0; i < testSize; i++ {
		x := grad.NewVar(xTest[i].(float64))
		y := grad.NewVar(yTest[i].(float64))

		xw := grad.Mul(x, w)
		r := grad.Sum(xw, b)

		prob := grad.Sigmoid(r)

		if math.Round(prob.Value()) == y.Value() {
			avgAcc += 1.0 / float64(testSize)
		}
	}

	fmt.Printf("Accuracy on the test set: %.4f\n", avgAcc)
}
