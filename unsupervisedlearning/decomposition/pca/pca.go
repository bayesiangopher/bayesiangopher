package pca

import (
	"errors"
	"github.com/bayesiangopher/bayesiangopher/core"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

var (
	CentringError = errors.New("ошибка центрирования данных - размерности")
	SVDError = errors.New("ошибка SVD разложения")
	ComponentsError = errors.New("компонент больше чем размерность данных")
)

type PCA struct{
	// Means is vector of means, for each random rector (col) in
	// train, for making centred data.
	Means			*mat.VecDense
	// Components is number of components after decomposition
	// (default - all).
	Components		int
	CentredTrain	*mat.Dense
	DecomposedTrain	*mat.Dense
	SVD				*mat.SVD
	Result			*mat.Dense
}

func (pca *PCA) Fit(train core.Train, components int) (err error) {
	// Create matrix from train
	M := core.MakeMatrixFromTrain(train)
	r, c := M.Dims()
	// Check components:
	if components == 0 { components = c}
	if components > c { panic(ComponentsError) }
	// Mean for centralization of data
	pca.Means = mat.NewVecDense((*train)[0].Elements, nil)
	for i := 0; i < c; i++ {
		col := mat.Col(nil, i, M)
		mean := stat.Mean(col, nil)
		pca.Means.SetVec(i, mean)
	}
	// Centre data
	pca.CentredTrain = centreData(M, pca.Means)
	// SVD decomposition of centredTrain:
	pca.SVD = &mat.SVD{}; ok := pca.SVD.Factorize(pca.CentredTrain, mat.SVDThin)
	if !ok { panic(SVDError) }
	var V mat.Dense; pca.SVD.VTo(&V)
	pca.Result = mat.NewDense(r, c, nil)
	pca.Result.Mul(pca.CentredTrain, &V)
	return
}

func centreData(M *mat.Dense, means *mat.VecDense) *mat.Dense {
	r, c := M.Dims()
	elements, _ := means.Dims()
	if c != elements { panic(CentringError) }
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			M.Set(i, j, M.At(i, j) - means.AtVec(j))
		}
	}
	return M
}


