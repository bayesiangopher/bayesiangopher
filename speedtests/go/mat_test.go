package speedtests

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

// TestCreateRandomMatrix testing speed and result of work
// createRandomMatrix function
func TestCreateRandomMatrix(t *testing.T) {
	for key, value := range mapMatTest {
		t.Run(key, testCreateRandomMatrixFunc(value[0], value[1]))
	}
}

func testCreateRandomMatrixFunc(N, M int) func(t *testing.T) {
	return func(t *testing.T) {
		A := createRandomMatrix(N, M)
		Rows, Cols := A.Dims()
		fmt.Printf("Created matrix 'A' with dimension: %d, %d \n", Rows, Cols); matPrint(A)
	}
}

// TestAdditionOfMatrices testing speed and result of work
// additionOfMatrices (gonum: Add) function
func TestAdditionOfMatrices(t *testing.T) {
	A := createRandomMatrix(16, 16)
	B := createRandomMatrix(16, 16)
	check := A.At(0, 0) + B.At(0, 0)
	C := additionOfMatrices(A, B)
	if C.At(0, 0) != check {
		t.Fatal("Матрицы сложились неправильно.")
	}
}

// TestSubtractOfMatrices testing speed and result of work
// suntractOfMatrices (gonum: Sub) function
func TestSubtractOfMatrices(t *testing.T) {
	A := createRandomMatrix(16, 16)
	B := createRandomMatrix(16, 16)
	check := A.At(0, 0) - B.At(0, 0)
	C := subtractOfMatrices(A, B)
	if C.At(0, 0) != check {
		t.Fatal("Вычитание матриц реализовано неправильно.")
	}
}

// TestScaleMatrix testing speed and result of work
// scaleMatrix (gonum: Scale) function
func TestScaleMatrix(t *testing.T) {
	A := createRandomMatrix(16, 16)
	check := A.At(0, 0) * 5.25
	scaleMatrix(5.25, A)
	if A.At(0, 0) != check {
		t.Fatal("Масштабирование матрицы реализовано неправильно.")
	}
}

// TestTransposeMatrix testing speed and result of work
// transposeMatrix (gonum: T) function
func TestTransposeMatrix(t *testing.T) {
	A := createRandomMatrix(16, 16)
	B := transposeMatrix(A)
	if A.At(2, 3) != B.At(3 , 2) {
		t.Fatal("Транспонирование матрицы реализовано неправильно.")
	}
}

// TestDotOfMatrices testing speed and result of work
// dotOfMatrices (gonum: Product) function
func TestDotOfMatrices(t *testing.T) {
	A := createRandomMatrix(16, 16)
	B := createRandomMatrix(16, 16)
	check := dotOfVectors(A.RowView(0), B.ColView(0))
	C := dotOfMatrices(A, B)
	if C.At(0, 0) != check {
		t.Fatal("Перемножение матрицы реализовано неправильно.")
	}
}

// TestDeterminantOfMatrix testing speed and result of work
// determinantOfMatrix (gonum: Det) function
func TestDeterminantOfMatrix(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 1, 1, 1})
	d := determinantOfMatrix(A)
	if d != 0.0 {
		t.Fatal("Нахождение определителя матрицы реализовано неправильно.")
	}
}

// TestEigensOfMatrix testing speed and result of work
// eigensOfMatrix (gonum: eig.Factorize) function
func TestEigensOfMatrix(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{
		1, -1,
		1, 1,
	})
	ev, _ := eigensOfMatrix(A)
	check := []complex128{1+1i, 1-1i}
	if ev[0] != check[0] || ev[1] != check[1] {
		t.Fatal("Собственные числа матрицы найдены неправильно.")
	}
}

// TestSVDOfMatrix testing speed and result of work
// SVDOfMatrix (gonum: SVD.Factorize) function
func TestSVDOfMatrix(t *testing.T) {
	A := createRandomMatrix(16, 16)
	S, U, V := SVDOfMatrix(A)
	sigma := mat.NewDense(16, 16, nil)
	for i := 0; i < 16; i++ {
		sigma.Set(i, i, S[i])
	}
	var ansFull mat.Dense
	ansFull.Product(U, sigma, V.T())
	if !mat.EqualApprox(&ansFull, A, 1e-8) {
		t.Errorf("SVD реализованно неправильно.")
	}
}

// TestCholeskyOfMatrix testing speed and result of work
// choleskyOfMatrix (gonum: cholesky.Factorize) function
func TestCholeskyOfMatrix(t *testing.T) {
	A := createRandomMatrix(16, 16)
	var AS mat.SymDense
	AS.SymOuterK(1, A)
	L := choleskyOfMatrix(&AS)
	var T mat.Dense
	T.Mul(L, L.T())
	if !mat.EqualApprox(&T, &AS, 1e-8) {
		t.Errorf("Разложение Холецкого реализованно неправильно.")
	}
}