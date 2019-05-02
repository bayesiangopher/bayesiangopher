// Set of wrappers for gonum vector algebra

package speedtests

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

const (
	VN = 1 << (iota * 10) // Sizes of vectors
	VNs
	VNm
	VNl
)

// createRandomVector create random N vector
func createRandomVector(N int) *mat.VecDense {
	v := mat.NewVecDense(N, nil)
	for i := 0; i < N; i++ {
		rand.Seed(int64(time.Now().UnixNano()))
		v.SetVec(i, rand.Float64() * 100)
	}
	return v
}

// scaleVector scale vector v
func scaleVector(v *mat.VecDense, alpha float64) {
	v.ScaleVec(alpha, v)
}

// frobeniusNormOfVector return Frobenius norm of vector
func frobeniusNormOfVector(v *mat.VecDense) (f float64) {
	f = mat.Norm(v, 2)
	return
}

// additionOfVectors return sum of vectors v, u,
// if alpha is not 0 return scaled sum of vectors (v + alpha * u)
func additionOfVectors(v, u *mat.VecDense, alpha float64) (w *mat.VecDense) {
	w = mat.NewVecDense(v.Len(), nil)
	if alpha == 0.0 {
		w.AddVec(u, v)
	}
	if alpha != 0.0 {
		w.AddScaledVec(v, alpha, u)
	}
	return
}

// subtractOfVectors return subtract of vectors v, u
func subtractOfVectors(v, u *mat.VecDense) (w *mat.VecDense) {
	w = mat.NewVecDense(v.Len(), nil)
	w.SubVec(v, u)
	return
}

// dotOfVectors return result of multiplication of two vectors
func dotOfVectors(v, u *mat.VecDense) (w float64) {
	w = mat.Dot(v, u)
	return
}

// vecPrint print vector in stdout
func vecPrint(v *mat.VecDense) {
	fmt.Printf("%v\n",
		mat.Formatted(v, mat.Prefix(" "), mat.Excerpt(3)))
}