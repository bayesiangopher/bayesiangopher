package dbscan

import (
	"fmt"
	"github.com/bayesiangopher/bayesiangopher/core"
	"testing"
)

// TestReadDataFromCSV
func TestReadDataFromCSV(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/the_real_tiny_dbscan_dataset.csv"}
	train := r.Read(true)
	fmt.Printf("\nFirst line of train: %v.\n", (*train)[0].Data)
	fmt.Printf("Second line of train: %v.\n", (*train)[1].Data)
	fmt.Printf("Last line of train: %v.\n", (*train)[len(*train) - 1].Data)
	fmt.Printf("Some metadata about train:" +
		"\nCount of rows: %v;" +
		"\nCount of elements in row: %v.\n\n",
		len(*train),
		(*train)[0].Elements)
}

// TestFit
func TestFit(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/the_real_tiny_dbscan_dataset.csv"}
	train := r.Read(true)
	dbscan := DBSCAN{}
	dbscan.Fit(train, 0.8, 10)
	fmt.Println("CLUSTERS:")
	for _, cluster := range dbscan.LabeledTrain {
		for _, vec := range cluster {
			core.VecPrint(&vec)
		}
		fmt.Println("==========")
	}
	fmt.Println("NOISE:")
	for _, vec := range dbscan.Noise {
		core.VecPrint(&vec)
	}

}
