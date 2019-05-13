package speedtests

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
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

// TestDotOfMatrices testing speed and result of work
// dotOfMatrices (gonum: Product) function
func TestDotOfMatrices(t *testing.T) {
	A := createRandomMatrix(16, 16)
	B := createRandomMatrix(16, 16)
	check := dotOfVectors(A.RowView(0), B.ColView(0))
	C := dotOfMatrices(A, B)
	if !almostEqual(check, C.At(0, 0), 1e-8) {
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

// BenchmarkCreateRandomMatrix testing speed and memory use of
// createRandomMatrix function
func BenchmarkCreateRandomMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchCreateRandomMatrixFunc(value[0], value[1]))
	}
}

func benchCreateRandomMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			A := createRandomMatrix(N, M)
			b.StopTimer()
			A.IsZero()
		}
	}
}

// BenchmarkScaleMatrix testing speed and memory use of
// scaleMatrix function
func BenchmarkScaleMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchScaleMatrixFunc(value[0], value[1], 5.25))
	}
}

func benchScaleMatrixFunc(N, M int, alpha float64) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			scaleMatrix(alpha, A)
			b.StopTimer()
			scaleMatrix(1 / alpha, A)
		}
	}
}

// BenchmarkTransposeMatrix testing speed and memory use of
// transposeMatrix function
func BenchmarkTransposeMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchTransposeMatrixFunc(value[0], value[1]))
	}
}

func benchTransposeMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			B := transposeMatrix(A)
			b.StopTimer()
			_ = B
		}
	}
}

// BenchmarkAdditionOfMatrix testing speed and memory use of
// additionOfMatrices function
func BenchmarkAdditionOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchAdditionOfMatrixFunc(value[0], value[1]))
	}
}

func benchAdditionOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		B := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			C := additionOfMatrices(A, B)
			b.StopTimer()
			_ = C
		}
	}
}

// BenchmarkSubtractOfMatrix testing speed and memory use of
// subtractOfMatrices function
func BenchmarkSubtractOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchSubtractOfMatrixFunc(value[0], value[1]))
	}
}

func benchSubtractOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		B := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			C := subtractOfMatrices(A, B)
			b.StopTimer()
			_ = C
		}
	}
}

// BenchmarkDotOfMatrix testing speed and memory use of
// dotOfMatrices function
func BenchmarkDotOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchDotOfMatrixFunc(value[0], value[1]))
	}
}

func benchDotOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		B := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			C := dotOfMatrices(A, B)
			b.StopTimer()
			_ = C
		}
	}
}

// BenchmarkDeterminantOfMatrix testing speed and memory use of
// determinantOfMatrix function
func BenchmarkDeterminantOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchDeterminantOfMatrixFunc(value[0], value[1]))
	}
}

func benchDeterminantOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			C := determinantOfMatrix(A)
			b.StopTimer()
			_ = C
		}
	}
}

// BenchmarkEigensOfMatrix testing speed and memory use of
// eigensOfMatrix function
func BenchmarkEigensOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchEigensOfMatrixFunc(value[0], value[1]))
	}
}

func benchEigensOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			ev, evec := eigensOfMatrix(A)
			b.StopTimer()
			_, _ = ev, evec
		}
	}
}

// BenchmarkSVDOfMatrix testing speed and memory use of
// SVDOfMatrix function
func BenchmarkSVDOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchSVDOfMatrixFunc(value[0], value[1]))
	}
}

func benchSVDOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			S, U, V := SVDOfMatrix(A)
			b.StopTimer()
			_, _, _ = S, U, V
		}
	}
}

// BenchmarkCholeskyOfMatrix testing speed and memory use of
// choleskyOfMatrix function
func BenchmarkCholeskyOfMatrix(b *testing.B) {
	for key, value := range mapMatTest {
		b.Run(key, benchCholeskyOfMatrixFunc(value[0], value[1]))
	}
}

func benchCholeskyOfMatrixFunc(N, M int) func(b *testing.B) {
	return func(b *testing.B) {
		A := createRandomMatrix(N, M)
		var AS mat.SymDense
		AS.SymOuterK(1, A)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			L := choleskyOfMatrix(&AS)
			b.StopTimer()
			_ = L
		}
	}
}

func almostEqual(a, b, eps float64) bool {
	return math.Abs(a - b) <= eps
}
