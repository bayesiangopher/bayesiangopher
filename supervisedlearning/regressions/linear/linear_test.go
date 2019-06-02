package linear

import (
	"errors"
	"fmt"
	"github.com/bayesiangopher/bayesiangopher/core"
	"testing"
)

// TestReadDataFromCSV - [OK]
func TestReadDataFromCSV(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/the_WWT_weather_10k_dataset.csv"}
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

// TestLRQR - [OK]
func TestLRQR(t *testing.T) {
	// train creation test:
	r := core.CSVReader{Path: "../../../datasets/the_WWT_weather_10k_dataset.csv"}
	train := r.Read(true)

	// QR method for linear regression test:
	standQR := LinearRegression(train, 0, QR)
	if err := standQR.Fit(); err != nil { t.Fatal("ошибка") }
	fmt.Printf("\nQR result: %v\n", standQR.parameterVector)

	rTest := core.CSVReader{Path: "../../../datasets/the_WWT_weather_test_no_y_train.csv"}
	trainTest := rTest.Read(true)
	result := standQR.Predict(trainTest, nil)
	core.VecPrint(result)

}

// TestDeterminationCoefficient - [OK]
func TestDeterminationCoefficient(t *testing.T) {
	// train creation test:
	r := core.CSVReader{Path: "../../../datasets/the_WWT_weather_10k_dataset.csv"}
	train := r.Read(true)
	rTest := core.CSVReader{Path: "../../../datasets/the_WWT_weather_test_train.csv"}
	trainTest := rTest.Read(true)

	// SVD method for linear regression test:
	standSVD := LinearRegression(train, 0, SVD)
	if err := standSVD.Fit(); err != nil { t.Fatal("ошибка") }
	fmt.Printf("\nQR result: %v\n", standSVD.parameterVector)
	fmt.Println(standSVD.DeterminationCoefficient(trainTest))
}

// TestLRSVD - [OK]
func TestLRSVD(t *testing.T) {
	// train creation test:
	r := core.CSVReader{Path: "../../../datasets/the_WWT_weather_10k_dataset.csv"}
	train := r.Read(true)

	// SVD method for linear regression test:
	standSVD := LinearRegression(train, 0, SVD)
	if err := standSVD.Fit(); err != nil { t.Fatal("ошибка") }
	fmt.Printf("\nQR result: %v\n", standSVD.parameterVector)

	rTest := core.CSVReader{Path: "../../../datasets/the_WWT_weather_test_no_y_train.csv"}
	trainTest := rTest.Read(true)
	result := standSVD.Predict(trainTest, nil)
	core.VecPrint(result)
}

// BenchmarkLRQR testing speed and memory use of
// lr.Fit(QR) function
func BenchmarkLRQR(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_WWT_weather_10k_dataset.csv"}
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
		LR := LinearRegression(train, targetColumn, method)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			err := LR.Fit()
			fmt.Println("fit succeed")
			b.StopTimer()
			if err != nil { panic(errors.New("fitting error")) }
		}
	}
}

// BenchmarkLRQR testing speed and memory use of
// lr.Fit(QR) function
func BenchmarkLRSVD(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_WWT_weather_10k_dataset.csv"}
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
		LR := LinearRegression(train, targetColumn, method)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			err := LR.Fit()
			fmt.Println("fit succeed")
			b.StopTimer()
			if err != nil { panic(errors.New("fitting error")) }
		}
	}
}