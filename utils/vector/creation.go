package vector

import (
	"math/rand"
	"time"
)

// Zero create and return zero vector N of float64.
func Zero(n int) *Vector {
	V := new(Vector)
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = 0.0
	}
	V.data = vector
	return V
}

// One create and return one vector N of float64.
func One(n, m int) *Vector {
	V := new(Vector)
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = 1.0
	}
	V.data = vector
	return V
}

// RandM create and return N vector with pseudo-random
// numbers filling by math/rand. Seed establishing by time epoch.
func RandM(n, upperbound int) *Vector {
	V := new(Vector)
	rand.Seed(time.Now().Unix())
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = rand.Float64() * float64(upperbound)
	}
	V.data = vector
	return V
}

// NormM create and return N vector with normal-distributed
// numbers filling by math/rand. Seed establishing by time epoch.
func NormM(n, stdDev, mean int) *Vector {
	V := new(Vector)
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = rand.NormFloat64()*float64(stdDev) + float64(mean)
	}
	V.data = vector
	return V
}
