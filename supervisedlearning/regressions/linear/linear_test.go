package linear

import (
	"fmt"
	"github.com/bayesiangopher/bayesiangopher/core"
	"testing"
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

func TestTinyFit(t *testing.T) {
	// train creation test:
	train := ReadDataFromCSV("../../../datasets/the_real_tiny_regression_dataset.csv")

	// QR method for linear regression test:
	var standQR LR
	if err := standQR.Fit(&train, 1, QR); err != nil { t.Fatal("ошибка") }
	fmt.Printf("\nQR result: %v\n", standQR.parameterVector)

	// SVD method for linear regression test:
	var standSVD LR
	if err := standSVD.Fit(&train, 1, SVD); err != nil { t.Fatal("ошибка") }
	fmt.Printf("SVD result: %v\n\n", standSVD.parameterVector)
}

func TestBostonFit(t *testing.T) {
	// train creation test:
	train := ReadDataFromCSV("../../../datasets/the_boston_housing_dataset.csv")

	// QR method for linear regression test:
	var standQR LR
	if err := standQR.Fit(&train, 9, QR); err != nil { t.Fatal("ошибка") }
	fmt.Printf("\nQR result: %v\n", standQR.parameterVector)

	// SVD method for linear regression test:
	var standSVD LR
	if err := standSVD.Fit(&train, 9, SVD); err != nil { t.Fatal("ошибка") }
	fmt.Printf("SVD result: %v\n\n", standSVD.parameterVector)
}