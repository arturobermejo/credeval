package dataframe

import (
	"encoding/csv"
	"io"
)

type Transform func([]any) error

type Dataframe struct {
	Data [][]any
}

func New(data ...[]any) *Dataframe {
	return &Dataframe{
		Data: data,
	}
}

func (df *Dataframe) Transform(colIndex int, transforms ...Transform) error {
	col := df.Data[colIndex]

	newCol := make([]any, len(col))
	copy(newCol, col)

	for _, tr := range transforms {
		err := tr(newCol)
		if err != nil {
			return err
		}
	}

	df.Data[colIndex] = newCol

	return nil
}

func LoadFromCSV(r io.Reader) (*Dataframe, error) {
	reader := csv.NewReader(r)

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) == 0 {
		return New(make([]any, 0)), nil
	}

	data := make([][]any, len(records[0]))

	for _, record := range records {
		for col, value := range record {
			data[col] = append(data[col], value)
		}
	}

	return New(data...), nil
}
