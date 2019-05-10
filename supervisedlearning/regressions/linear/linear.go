package linear

import (
	"errors"
	"fmt"
	"github.com/bayesiangopher/bayesiangopher/core"
	"gonum.org/v1/gonum/mat"
	"log"
)

var (
	SVDDecompositionError = errors.New("ошибка во время SVD разложения")
	SVDResultComputeError = errors.New("вектор b посчитан неправильно")
	FittingError = errors.New("ошибка обучения")
)

// LRtype specifies the treatment of solving
type LRtype int

const (
	// QR specifies solving with QR decomposition
	QR LRtype = 1 << (iota + 1)
	// SVD specifies solving with SVD decomposition
	SVD
)

// LinearRegression is main struct for linear regression package
type LR struct {
	regressand		*mat.VecDense
	regressors		*mat.Dense
	parameterVector	*mat.VecDense
	method 			LRtype
}

// Fit train to LR
func (lr *LR) Fit(train core.Train, targetColumn int, method LRtype) (err error) {
	// Prepare structure and
	// Prepare data:
	lr.method = method
	lr.regressors = mat.NewDense(len(*train), (*train)[0].Elements - 1, nil)
	fmt.Println(lr.regressors.Dims())
	lr.regressand = mat.NewVecDense(len(*train), nil)
	for index, row := range *train {
		for idx, element := range row.Data {
			if idx == targetColumn {
				lr.regressand.SetVec(index, element)
			} else {
				if idx > targetColumn { idx -= 1}
				lr.regressors.Set(index, idx, element)
			}
		}
	}
	// Make fitting:
	switch {
	case method&QR != 0:
		lr.parameterVector = qrRegressionSolver(lr.regressors, lr.regressand)
	case method&SVD != 0:
		lr.parameterVector, err = svdRegressionSolver(lr.regressors, lr.regressand)
		if err != nil { log.Fatal(err) }
	}
	return nil
}

func qrRegressionSolver(A *mat.Dense, y *mat.VecDense) (b *mat.VecDense){
	_, c := A.Dims()
	b = mat.NewVecDense(c, nil)
	// QR decomposition is often used to solve the linear least squares problem
	// b = R' * QT * y ~ R * b = QT * y => give us b.
	// Q -  orthogonal matrix m x n;
	// R -  upper triangular matrix n x n.
	// (see - https://en.wikipedia.org/wiki/QR_decomposition)
	var QR mat.QR
	var Q, R, Qt, Qty mat.Dense
	QR.Factorize(A); QR.QTo(&Q); QR.RTo(&R); Qt.Clone(Q.T()); Qty.Mul(&Qt, y)
	// Now find b:
	for i := c - 1; i >= 0; i-- {
		b.SetVec(i, Qty.At(i, 0))
		for j := i + 1; j < c; j++ {
			b.SetVec(i, b.AtVec(i) - b.AtVec(j) * R.At(i, j))
		}
		b.SetVec(i, b.AtVec(i) / R.At(i, i))
	}
	return
}

func svdRegressionSolver(A *mat.Dense, y *mat.VecDense) (b *mat.VecDense, err error) {
	r, c := A.Dims()
	b = mat.NewVecDense(c, nil)
	// SVD decomposition is often used to solve the linear least squares problem
	// b =  X+ * y, where X+ = U * D+ * VT, where:
	// D+ - generalized inverse of E, where
	// E - is a diagonal m × n matrix with non-negative real numbers on the diagonal;
	// U - is an m × m unitary matrix;
	// VT - transposed n × n unitary matrix.
	// (see - https://en.wikipedia.org/wiki/Singular_value_decomposition)
	var SVD mat.SVD
	var V, U mat.Dense
	SVD.Factorize(A, mat.SVDFull); SVD.VTo(&V); SVD.UTo(&U); container := SVD.Values(nil)
	if vr, vc := V.Dims(); vr != c && vc != c { return nil, SVDDecompositionError }
	if ur, uc := U.Dims(); ur != r && uc != r { return nil, SVDDecompositionError }
	D := mat.NewDense(r, c, nil)
	for idx, element := range container {
		D.Set(idx, idx, 1 / element)
	}
	// Now find b:
	var Vt, DVt, UDVt mat.Dense
	Vt.Clone(V.T())
	DVt.Mul(D, &Vt)
	UDVt.Mul(&U, &DVt)
	var temp mat.Dense
	temp.Mul(UDVt.T(), y)
	if _, c := temp.Dims(); c != 1 { return nil, SVDResultComputeError }
	view := temp.ColView(0)
	for i := 0; i < view.Len(); i++ {
		b.SetVec(i, view.AtVec(i))
	}
	return
}

