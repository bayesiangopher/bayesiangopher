package pca

import (
	"fmt"
	"log"
	"testing"
	"github.com/bayesiangopher/bayesiangopher/core"
)

// TestReadDataFromCSV
func TestReadDataFromCSV(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/the_real_tiny_pca_dataset.csv"}
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
	r := core.CSVReader{Path: "../../../datasets/the_real_tiny_pca_dataset.csv"}
	train := r.Read(true)
	pca := PCA{}
	pca.Fit(train, 0)
	if pca.Means.AtVec(0) != 5.5 || pca.Means.AtVec(1) != 10.825535848000001 {
		log.Fatal("Mistake in means computing.")
	}
	fmt.Printf("Mean vector: %v \n", pca.Means)
	if pca.CentredTrain.At(0,0) != (*train)[0].Data[0] - pca.Means.At(0,0) {
		log.Fatal("Centring mistake.")
	}
	fmt.Printf("Centred data: %v \n", pca.CentredTrain)
	if pca.Result.At(0, 0) != -9.250319037967504 {
		log.Fatal("Result mistake.")
	}
	fmt.Printf("Result of decomposition: %v \n", pca.Result)
	_ = pca.Losses()
}