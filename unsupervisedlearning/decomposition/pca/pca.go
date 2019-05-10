package pca

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"github.com/bayesiangopher/bayesiangopher/core"
)

var ()

type PCA struct{
	// Components is number of components after decomposition
	// (default - all).
	Components		int
	SVD				*mat.SVD
}

func (pca *PCA) Fit(train *[]Row, components int) (err error) {
	// Create matrix from train
	M := MakeMatrixFromTrain(train)
	r, c := M.Dims()
	// Mean for centralization of data
	v := mat.NewVecDense(train[0].elements, nil)
	for i := 0; i < c; i++ {
		col := mat.Col(nil, i, M)
		mean := stat.Mean(col, nil)
		v.SetVec(i, mean)
	}
	return
}



