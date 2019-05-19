package pca

import (
	"errors"
	"fmt"
	"github.com/bayesiangopher/bayesiangopher/core"
	"log"
	"testing"
)

// TestReadDataFromCSV
func TestReadDataFromCSV(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/the_boston_housing_dataset.csv"}
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
	r := core.CSVReader{Path: "../../../datasets/the_boston_housing_dataset.csv"}
	train := r.Read(true)
	pca := PCA{}
	err := pca.Fit(train, 0)
	if err != nil { log.Fatal(errors.New("ошибка фита")) }
	fmt.Println(pca.Result)

}

// BenchmarkLRQR testing speed and memory use of
// lr.Fit(QR) function
func BenchmarkFit(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_boston_housing_dataset.csv"}
	train := r.Read(true)
	pca := PCA{}
	b.ReportAllocs()
	b.N = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StartTimer()
		err := pca.Fit(train, 0)
		if err != nil { log.Fatal(errors.New("ошибка фита")) }
		b.StopTimer()
	}
}
