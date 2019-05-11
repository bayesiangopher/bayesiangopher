package core

import (
	"errors"
	"gonum.org/v1/gonum/mat"
	"math"
)

var (
	DimError = errors.New("не совпадают размерности")
)

// VecEuclidean == L2
func VecEuclidean(v, u *mat.VecDense) (dist float64) {
	CheckDim(v, u)
	vr, _ := v.Dims()
	for i := 0; i < vr; i++ {
		dist += math.Pow(v.AtVec(i) - u.AtVec(i), 2)
	}
	return math.Sqrt(dist)
}

// Taxicab geometry == L1
func VecL1(v, u *mat.VecDense) (dist float64) {
	CheckDim(v, u)
	vr, _ := v.Dims()
	for i := 0; i < vr; i++ {
		dist += math.Abs(v.AtVec(i) - u.AtVec(i))
	}
	return
}

// Canberra distance
func VecCanberra(v, u *mat.VecDense) (dist float64) {
	CheckDim(v, u)
	vr, _ := v.Dims()
	for i := 0; i < vr; i++ {
		numerator := math.Abs(v.AtVec(i) - u.AtVec(i))
		denominator := math.Abs(v.AtVec(i)) + math.Abs(u.AtVec(i))
		dist += numerator / denominator
	}
	return
}

// Chebyshev distance
func VecChebyshev(v, u *mat.VecDense) (dist float64) {
	CheckDim(v, u)
	vr, _ := v.Dims()
	for i := 0; i < vr; i++ {
		dist = math.Max(dist, math.Abs(v.AtVec(i) - u.AtVec(i)))
	}
	return
}

func CheckDim(N, M mat.Matrix) {
	Nr, Nc := N.Dims()
	Mr, Mc := M.Dims()
	switch {
	case Nc == 1 && Mc == 1: if Nr != Mr { panic(DimError) }
	default: if Nc != Mr { panic(DimError) }
	}
}