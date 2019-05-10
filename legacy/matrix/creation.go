package matrix

import (
	"math/rand"
	"time"
)

// Zero create and return zero matrix NxM of float64.
func Zero(n, m int) *Matrix {
	M := new(Matrix)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, m)
		for j := range matrix[i] {
			matrix[i][j] = 0.0
		}
	}
	M.data = matrix
	return M
}

// One create and return one matrix NxM of float64.
func One(n, m int) *Matrix {
	M := new(Matrix)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, m)
		for j := range matrix[i] {
			if i == j {
				matrix[i][j] = 1.0
			} else {
				matrix[i][j] = 0.0
			}
		}
	}
	M.data = matrix
	return M
}

// RandM create and return NxM matrix with pseudo-random
// numbers filling by math/rand. Seed establishing by time epoch.
func RandM(n, m, upperbound int) *Matrix {
	M := new(Matrix)
	rand.Seed(time.Now().Unix())
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, m)
		for j := range matrix[i] {
			matrix[i][j] = rand.Float64() * float64(upperbound)
		}
	}
	M.data = matrix
	return M
}

// NormM create and return NxM matrix with normal-distributed
// numbers filling by math/rand. Seed establishing by time epoch.
func NormM(n, m, stdDev, mean int) *Matrix {
	M := new(Matrix)
	rand.Seed(time.Now().Unix())
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, m)
		for j := range matrix[i] {
			matrix[i][j] = rand.NormFloat64()*float64(stdDev) + float64(mean)
		}
	}
	M.data = matrix
	return M
}
