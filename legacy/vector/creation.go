package vector

import (
	"math/rand"
	"time"
)

// Zero creates and returns zero filled vector N of float64.
func Zero(n int) *Vector {
	V := new(Vector)
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = 0.0
	}
	V.n, V.data = n, vector
	return V
}

// One creates and returns one filled vector N of float64.
func One(n int) *Vector {
	V := new(Vector)
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = 1.0
	}
	V.n, V.data = n, vector
	return V
}

// RandM creates and returns N vector filled with pseudo-random
// numbers from math/rand. Seed establishing by time epoch.
func RandM(n, upperbound int) *Vector {
	V := new(Vector)
	rand.Seed(time.Now().Unix())
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = rand.Float64() * float64(upperbound)
	}
	V.n, V.data = n, vector
	return V
}

// NormM create and return N vector filled with normal-distributed
// numbers from math/rand. Seed establishing by time epoch.
func NormM(n, stdDev, mean int) *Vector {
	V := new(Vector)
	vector := make([]float64, n)
	for i := range vector {
		vector[i] = rand.NormFloat64()*float64(stdDev) + float64(mean)
	}
	V.n, V.data = n, vector
	return V
}
