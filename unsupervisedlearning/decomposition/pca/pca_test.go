package pca

import (
	"fmt"
	"testing"
	"github.com/bayesiangopher/bayesiangopher/core"
)

// TestReadDataFromCSV - [OK]
func TestReadDataFromCSV(t *testing.T) {
	train := core.ReadDataFromCSV("../../../datasets/the_real_tiny_regression_dataset.csv")
	fmt.Printf("\nFirst line of train: %v.\n", train[0].data)
	fmt.Printf("Second line of train: %v.\n", train[1].data)
	fmt.Printf("Last line of train: %v.\n", train[len(train) - 1].data)
	fmt.Printf("Some metadata about train:" +
		"\nCount of rows: %v;" +
		"\nCount of elements in row: %v.\n\n",
		len(train),
		train[0].elements)
}
