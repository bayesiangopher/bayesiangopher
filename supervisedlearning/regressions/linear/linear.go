package linear

import (
	"errors"
	"github.com/bayesiangopher/bayesiangopher/core"
	"gonum.org/v1/gonum/mat"
	"log"
	"math"
)

var (
	SVDDecompositionError = errors.New("ошибка во время SVD разложения")
	SVDResultComputeError = errors.New("вектор b посчитан неправильно")
	CoefBeforeFitting = errors.New("поиск коэффициентов до фиттинга")
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
	targetColumn 	int
}

// Fit train to LR
func (lr *LR) Fit(train core.Train, targetColumn int, method LRtype) (err error) {
	// Prepare structure and
	// Prepare data:
	lr.method = method
	lr.targetColumn = targetColumn
	lr.regressors = mat.NewDense(len(*train), (*train)[0].Elements - 1, nil)
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
	default:
		lr.parameterVector, err = svdRegressionSolver(lr.regressors, lr.regressand)
		if err != nil { log.Fatal(err) }
	}
	return nil
}

// Predict does predict
func (lr *LR) Predict(testTrain core.Train) (predictResult *mat.VecDense) {
	testTrainMatrix := core.MakeMatrixFromTrain(testTrain)
	r, c := testTrainMatrix.Dims()
	predictResult = mat.NewVecDense(r, nil)
	var result float64
	for i := 0; i < r; i++ {
		result = 0
		for j := 0; j < c; j++ {
			result += testTrainMatrix.At(i,j) * lr.parameterVector.AtVec(j)
		}
		predictResult.SetVec(i, result)
	}
	return
}

// TrainCoef checks result of fitting
func (lr *LR) TrainCoef() (coef float64) {
	r, c := lr.regressors.Dims()
	if lr.parameterVector == nil {
		panic( CoefBeforeFitting )
	}
	coefs := make([]float64, r)
	var resultCheck float64
	for i := 0; i < r; i++ {
		resultCheck = 0
		for j := 0; j < c; j++ {
			resultCheck += lr.regressors.At(i, j) * lr.parameterVector.AtVec(j)
		}
		coefs[i] = math.Abs(lr.regressand.AtVec(i) / resultCheck)
	}
	min, _ := Min(coefs)
	return min
}

// TestCoef checks result of fitting for test train data
func (lr *LR) TestCoef(testTrain core.Train) (coef float64) {
	regressors := mat.NewDense(len(*testTrain), (*testTrain)[0].Elements - 1, nil)
	regressand := mat.NewVecDense(len(*testTrain), nil)
	for index, row := range *testTrain {
		for idx, element := range row.Data {
			if idx == lr.targetColumn {
				regressand.SetVec(index, element)
			} else {
				if idx > lr.targetColumn { idx -= 1}
				regressors.Set(index, idx, element)
			}
		}
	}
	r, c := regressors.Dims()
	coefs := make([]float64, r)
	var resultCheck float64
	for i := 0; i < r; i++ {
		resultCheck = 0
		for j := 0; j < c; j++ {
			resultCheck += regressors.At(i, j) * lr.parameterVector.AtVec(j)
		}
		coefs[i] = math.Abs(regressand.AtVec(i) / resultCheck)
	}
	min, _ := Min(coefs)
	return min
}

func Min(values []float64) (min float64, err error) {
	if len(values) == 0 {
		return 0, errors.New("пустой слайс")
	}
	min = values[0]
	for _, v := range values {
		if v < min { min = v }
	}
	return min, nil
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
	QR.Factorize(A)
	QR.QTo(&Q)
	QR.RTo(&R)
	Qt.Clone(Q.T())
	Qty.Mul(&Qt, y)
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
	SVD.Factorize(A, mat.SVDFull)
	SVD.VTo(&V)
	SVD.UTo(&U)
	container := SVD.Values(nil)
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
