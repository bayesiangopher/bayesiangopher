package speedtests

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
)

const (
	N = 1000    // Len of rows
	M = 1000    // Len of columns
	K = 1000000 // Len of vectors
)

func matTests() {

	// In memory creating Matrix's elements:
	v := make([]float64, N*M)
	for i := 0; i < N*M; i++ {
		rand.Seed(int64(i))
		v[i] = rand.Float64() * 100
	}

	// Create a new Matrix:
	A := mat.NewDense(N, M, v)
	fmt.Println("A:")
	matPrint(A)

	// Setting and getting functions:
	a := A.At(0, 2)
	fmt.Println("A[0, 2]:", a)

	A.Set(0, 2, -1.5)
	a = A.At(0, 2)
	fmt.Println("A[0, 2]:", a)

	// Extracting columns and rows:
	fmt.Println("Row 1 of A:")
	matPrint(A.RowView(0))
	fmt.Println("Column 2 of A:")
	matPrint(A.ColView(1))

	// Set rows and columns:
	row := make([]float64, M)
	for i := 0; i < M; i++ {
		rand.Seed(int64(i + M))
		row[i] = rand.Float64() * 100
	}
	A.SetRow(0, row)
	matPrint(A)
	column := make([]float64, N)
	for i := 0; i < N; i++ {
		rand.Seed(int64(i + N*2))
		column[i] = rand.Float64() * 100
	}
	A.SetCol(0, column)
	matPrint(A)

	// Addition of Matrices:
	B := mat.NewDense(N, M, nil)
	B.Add(A, A)
	fmt.Println("B: ")
	matPrint(B)

	// Subtractions of Matrices:
	C := mat.NewDense(N, M, nil)
	C.Sub(B, A)
	fmt.Println("B - A:")
	matPrint(C)

	// Scaling:
	C.Scale(3.5, B)
	fmt.Println("3.5 * B:")
	matPrint(C)

	// Transposing:
	// Now we cant set any of A.T() values.
	fmt.Println("A'")
	matPrint(A.T())

	// Multiplication:
	D := mat.NewDense(N, M, nil)
	D.Product(A, B)
	fmt.Println("A * B:")
	matPrint(D)

	// Product use for multiply any matrices:
	D.Product(D, A, B.T(), D, D, D)
	fmt.Println("D * A * B' * D * D * D")
	matPrint(D)

	// Custom functions:
	D.Apply(sumOfIndices, A)
	fmt.Println("D:")
	matPrint(D)

	// Determinant:
	E := A.Slice(0, 5, 0, 5)
	d := mat.Det(E)
	fmt.Printf("Determinant of A[0, 5][0, 5] is: %e\n", d)

	// Trace:
	t := mat.Trace(E)
	fmt.Printf("Trace of A[0, 5][0, 5]: %e\n", t)

	// Init Matrix another way (as product of 2 matrices):
	var G mat.Dense
	G.Mul(A, A)
	matPrint(&G)
	G.Reset()
	fmt.Printf("Is G zero matrix: %t", G.IsZero())

	// Eigen:
	var eig mat.Eigen
	ok := eig.Factorize(A, mat.EigenBoth)
	if !ok {
		log.Fatal("Eigendecomposition failed")
	}
	fmt.Printf("Eigenvalues of A:\n%v\n", eig.Values(nil))
	ev := eig.VectorsTo(nil)
	r, c := ev.Dims()
	fmt.Printf("Dim of Eigenvectors matrix of A: %d, %d\n", r, c)

	// SVD (Singular Value Decomposition):
	var svd mat.SVD
	// Full because N == M
	ok = svd.Factorize(A, mat.SVDFull)
	if !ok {
		log.Fatal("SVD failed")
	}
	fmt.Printf("Singular values of A:\n%v\n", svd.Values(nil))
	fmt.Println("U (mxm orthogonal matrix:")
	matPrint(svd.UTo(nil))
	fmt.Println("V (nxn orthogonal matrix:")
	matPrint(svd.VTo(nil))

	// Invertible matrix:
	kek := mat.NewDense(2, 2, []float64{
		4, 0,
		0, 4,
	})
	var ia mat.Dense
	err := ia.Inverse(kek)
	if err != nil {
		log.Fatal("Inverse failed")
	}
	matPrint(kek)
	fmt.Println("Result of Inverse check:")
	var l mat.Dense
	l.Mul(kek, &ia)
	matPrint(&l)

	// Cholesky decomposition:
	var AS mat.SymDense
	AS.SymOuterK(1, A)
	fmt.Println("AS (symmetric):")
	matPrint(&AS)
	var chol mat.Cholesky
	if ok = chol.Factorize(&AS); !ok {
		log.Fatal("A matrix is not positive semi-definite.")
	}
	L := chol.LTo(nil)
	var test mat.Dense
	test.Mul(L, L.T())
	fmt.Println("AS:")
	matPrint(&AS)
	fmt.Println("Result of check (must be equal AS):")
	matPrint(&test)

}

// matPrint print Matrix to Stdout
func matPrint(A mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(A, mat.Prefix(" "), mat.Excerpt(3)))
}

// sumOfIndices return sum of two indices
func sumOfIndices(i, j int, v float64) float64 {
	return float64(i + j)
}
