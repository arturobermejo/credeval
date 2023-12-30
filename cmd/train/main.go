package main

import (
	"flag"
	"io"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/arturobermejo/credeval/dataframe"
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
	filename := flag.String("f", "dataset.csv", "dataset csv file")
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
}
