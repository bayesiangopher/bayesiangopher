package linear

import (
	"fmt"
	"github.com/bayesiangopher/bayesiangopher/core"
	"github.com/pkg/errors"
	"testing"
)

// TestReadDataFromCSV - [OK]
func TestReadDataFromCSV(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/the_train_WWT_weather_dataset.csv"}
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

func TestLRQR(t *testing.T) {
	// train creation test:
	r := core.CSVReader{Path: "../../../datasets/the_train_WWT_weather_dataset.csv"}
	train := r.Read(true)

	// QR method for linear regression test:
	var standQR LR
	if err := standQR.Fit(train, 0, QR); err != nil { t.Fatal("ошибка") }
	fmt.Printf("\nQR result: %v\n", standQR.parameterVector)
	fmt.Printf("TrainCoef: %v\n", standQR.TrainCoef())

	// test train:
	r = core.CSVReader{Path: "../../../datasets/the_test_train_WWT_weather_dataset.csv"}
	train = r.Read(true)
	fmt.Printf("TestCoef: %v\n", standQR.TestCoef(train))
}

// BenchmarkLRQR testing speed and memory use of
// lr.Fit(QR) function
func BenchmarkLRQR(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_train_WWT_weather_dataset.csv"}
	train := r.Read(true)
	targetColumn := 0
	count := 0
	for count < 1 {
		b.Run("BenchmarkLRQR", benchLRQR(train, targetColumn, QR))
		count += 1
	}
}

func benchLRQR(train core.Train, targetColumn int, method LRtype) func(b *testing.B) {
	return func(b *testing.B) {
		var standQR LR
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			err := standQR.Fit(train, targetColumn, method)
			fmt.Println("fit succeed")
			b.StopTimer()
			if err != nil { panic(errors.New("fitting error")) }
		}
	}
}

func TestLRSVD(t *testing.T) {
	// train creation test:
	r := core.CSVReader{Path: "../../../datasets/the_train_WWT_weather_dataset.csv"}
	train := r.Read(true)

	// SVD method for linear regression test:
	var standSVD LR
	if err := standSVD.Fit(train, 0, SVD); err != nil { t.Fatal("ошибка") }
	fmt.Printf("SVD result: %v\n\n", standSVD.parameterVector)
	fmt.Printf("TrainCoef: %v\n", standSVD.TrainCoef())

	// test train:
	r = core.CSVReader{Path: "../../../datasets/the_test_train_WWT_weather_dataset.csv"}
	train = r.Read(true)
	fmt.Printf("TestCoef: %v\n", standSVD.TestCoef(train))
}

// BenchmarkLRQR testing speed and memory use of
// lr.Fit(QR) function
func BenchmarkLRSVD(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_train_WWT_weather_dataset.csv"}
	train := r.Read(true)
	targetColumn := 0
	count := 0
	for count < 1 {
		b.Run("BenchmarkLRSVD", benchLRSVD(train, targetColumn, SVD))
		count += 1
	}
}

func benchLRSVD(train core.Train, targetColumn int, method LRtype) func(b *testing.B) {
	return func(b *testing.B) {
		var standSVD LR
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			err := standSVD.Fit(train, targetColumn, method)
			fmt.Println("fit succeed")
			b.StopTimer()
			if err != nil { panic(errors.New("fitting error")) }
		}
	}
}