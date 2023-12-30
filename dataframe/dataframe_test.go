package dataframe

import (
	"io"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDataFrame(t *testing.T) {
	assert := assert.New(t)

	df := New([]any{1.0, 2.0}, []any{3.0, 4.0})

	expected := [][]any{{1.0, 2.0}, {3.0, 4.0}}

	assert.ElementsMatch(expected, df.Data)
}

func TestLoadFromCSV(t *testing.T) {
	assert := assert.New(t)

	file, err := os.Open("samples/example.csv")
	defer file.Close()

	assert.NoError(err)

	df, err := LoadFromCSV(io.Reader(file))
	assert.NoError(err)

	expected := [][]any{
		{"col1", "0.34", "0.45", "3"},
		{"col2", "0.45", "1.45", "4"},
		{"col3", "0.67", "34.0", "678"},
	}

	assert.ElementsMatch(expected, df.Data)
}

func TestTransform(t *testing.T) {
	assert := assert.New(t)

	df := New([]any{1.0, 2.0}, []any{3.0, 4.0})

	transform := func(col []any) error {
		for i, v := range col {
			col[i] = v.(float64) + 1.0
		}

		return nil
	}

	df.Transform(1, transform)

	expected := [][]any{{1.0, 2.0}, {4.0, 5.0}}

	assert.ElementsMatch(expected, df.Data)
}
