package core

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func recoverDim() {
	if r := recover(); r != nil {
		fmt.Println("recovered from", r)
	}
}

func TestCheckDimVec(t *testing.T) {
	defer recoverDim()
	buf := []float64{0,1,2,3,4,5,6,7,8,9}
	bufOk := []float64{0,1,2,3,4,5,6,7,8,9}
	bufFail := []float64{0,1,2,3,4,5,6,7}
	vec := mat.NewVecDense(len(buf), buf)
	VecPrint(vec)
	vecOk := mat.NewVecDense(len(bufOk), bufOk)
	VecPrint(vecOk)
	vecFail := mat.NewVecDense(len(bufFail), bufFail)
	VecPrint(vecFail)
	CheckDim(vec, vecOk)
	fmt.Println("1st test passed.")
	CheckDim(vec, vecFail)
}

func TestCheckDimMat(t *testing.T) {
	defer recoverDim()
	buf := []float64{0,1,2,3,4,5,6,7,8,9,10,11}
	bufOk := []float64{0,1,2,3,4,5}
	bufFail := []float64{0,1,2,3,4,5,6,7}
	matx := mat.NewDense(2, len(buf) / 2, buf)
	MatPrint(matx)
	matOk := mat.NewDense(len(bufOk), 1, bufOk)
	MatPrint(matOk)
	matFail := mat.NewDense(2, len(bufFail) / 2, bufFail)
	MatPrint(matFail)
	CheckDim(matx, matOk)
	fmt.Println("1st test passed.")
	CheckDim(matx, matFail)
}
