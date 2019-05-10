package vector

import (
	"fmt"
	"math"
)

func check_dim(n, m int) error {
	if n != m {
		return fmt.Errorf("Vectors must be same size") //needs error handler
	}
	return nil
}

func L1(v1, v2 *Vector) (float64, error) {
	if check_dim(v1.n, v2.n) != nil {
		panic("Vectors must be same size")
	}
	var dist float64
	for i := range v1.data {
		dist += math.Abs(v1.data[i] - v2.data[i])
	}
	return dist, nil
}

func L2(v1, v2 *Vector) (float64, error) {
	if check_dim(v1.n, v2.n) != nil {
		panic("Vectors must be same size")
	}
	var dist float64
	for i := range v1.data {
		dist += (v1.data[i] - v2.data[i]) * (v1.data[i] - v2.data[i])
	}
	return math.Sqrt(dist), nil
}

func CanberraDistance(v1, v2 *Vector) (float64, error) {
	if check_dim(v1.n, v2.n) != nil {
		panic("Vectors must be same size")
	}
	var dist float64
	for i := range v1.data {
		dist += (math.Abs(v1.data[i]-v2.data[i]) / (math.Abs(v1.data[i]) + math.Abs(v2.data[i])))
	}
	return dist, nil
}

func ChebyshevDistance(v1, v2 *Vector) (float64, error) {
	if check_dim(v1.n, v2.n) != nil {
		panic("Vectors must be same size")
	}
	var dist float64
	for i := range v1.data {
		dist = math.Max(dist, math.Abs(v1.data[i]-v2.data[i]))
	}
	return dist, nil
}
