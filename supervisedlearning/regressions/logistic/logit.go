package logit

import (
	"errors"
	"math"
	"math/rand"

	"github.com/bayesiangopher/bayesiangopher/core"
	"gonum.org/v1/gonum/mat"
)

var (
	BatchSizeInputError = errors.New("Batch size exceeds the size of data")
)

type LGRsolver int

const (
	// Gradient Descending solver
	GD LGRsolver = 1 << iota
	// Stochastic Gradient Descending solver
	SGD
)

type LGR struct {
	bias        *mat.VecDense //bias; default: random vetor
	lrate       float64       //learning rate
	iter        int           //number of iteration
	batchs      int           //batch size
	weights     *mat.VecDense
	solver      LGRsolver //solving method: GD, SGD
	multi_class bool
	res         *mat.Dense
}

func (lgr *LGR) Fit(train core.Train, targetColumn int) (err error) {
	// Prepare data:
	x_train := mat.NewDense(len(*train), (*train)[0].Elements-1, nil)
	y_train := mat.NewVecDense(len(*train), nil)
	for index, row := range *train {
		for idx, elem := range row.Data {
			if idx == targetColumn {
				y_train.SetVec(index, elem)
			} else {
				if idx > targetColumn {
					idx -= 1
				}
				x_train.Set(index, idx, elem)
			}
		}
	}

	xrn, xcn := x_train.Dims()
	//Set optinal parametrs
	if lgr.bias == nil {
		rbias := make([]float64, xrn)
		for i := range rbias {
			rbias[i] = rand.NormFloat64()
		}
		lgr.bias = mat.NewVecDense(len(rbias), rbias)
	}
	if lgr.lrate == 0.0 {
		lgr.lrate = 0.01
	}
	if lgr.iter == 0 {
		lgr.iter = 10000
	}
	if lgr.batchs == 0 {
		lgr.batchs = int(math.Round(float64(xrn) / 5.0))
	} else if lgr.batchs >= xrn {
		return BatchSizeInputError
	}
	if lgr.weights == nil {
		rweights := make([]float64, xcn)
		for i := range rweights {
			rweights[i] = rand.NormFloat64()
		}
		lgr.weights = mat.NewVecDense(len(rweights), rweights)
	}

	switch {
	case lgr.solver&GD != 0:
		gdSolver(x_train, y_train, lgr.weights, lgr.bias, lgr.lrate, lgr.iter)
	case lgr.solver&SGD != 0:
		sgdSolver(x_train, y_train, lgr.weights, lgr.bias, lgr.lrate, lgr.iter, lgr.batchs)
	default:
		sgdSolver(x_train, y_train, lgr.weights, lgr.bias, lgr.lrate, lgr.iter, lgr.batchs)

	}
	return nil
}

func (lgr *LGR) Predict(train core.Train, w *mat.VecDense) {
	b := mat.NewVecDense(len(*train), nil)

	x_test := mat.NewDense(len(*train), (*train)[0].Elements-1, nil)
	for index, row := range *train {
		for idx, elem := range row.Data {
			if idx != 0 {
				if idx > 0 {
					idx -= 1
				}
				x_test.Set(index, idx, elem)
			}
		}
	}

	lgr.res = mat.NewDense(len(*train), 1, nil)
	sigmoid(x_test, w, b, lgr.res)
}

func gdSolver(X *mat.Dense, y *mat.VecDense, w *mat.VecDense, b *mat.VecDense, lr float64, iter int) {
	h := mat.NewDense(y.Len(), 1, nil)
	grad := mat.NewVecDense(w.Len(), nil)
	for i := 0; i < iter; i++ {
		sigmoid(X, w, b, h)
		subg := mat.NewVecDense(y.Len(), nil)
		subg.SubVec(y, h.ColView(0))
		grad.MulVec(X.T(), subg)
		grad.ScaleVec(1/(float64)(y.Len()), grad)
		grad.ScaleVec(-lr, grad)
		w.SubVec(grad, w)
	}
}

func sgdSolver(X *mat.Dense, y *mat.VecDense, w *mat.VecDense, b *mat.VecDense, lr float64, iter int, batch int) {
	h := mat.NewDense(batch, 1, nil)
	grad := mat.NewVecDense(w.Len(), nil)
	epoch := int(math.Round(float64(y.Len()) / float64(batch)))
	for i := 0; i < iter; i++ {
		for j := 0; j < epoch; j++ {
			rnd_idx := rand.Intn(y.Len() - batch)
			xb := X.Slice(rnd_idx, rnd_idx+batch, 0, w.Len())
			Xb := mat.DenseCopyOf(xb)
			y_b := y.SliceVec(rnd_idx, rnd_idx+batch)
			yb := mat.VecDenseCopyOf(y_b)
			bias_b := y.SliceVec(rnd_idx, rnd_idx+batch)
			bias := mat.VecDenseCopyOf(bias_b)
			sigmoid(Xb, w, bias, h)
			subg := mat.NewVecDense(yb.Len(), nil)
			subg.SubVec(yb, h.ColView(0))
			grad.MulVec(Xb.T(), subg)
			grad.ScaleVec(1/(float64)(yb.Len()), grad)
			grad.ScaleVec(-lr, grad)
			w.SubVec(grad, w)
		}
	}
}

func sigmoid(X *mat.Dense, w *mat.VecDense, b *mat.VecDense, h *mat.Dense) {
	sigm := func(_, _ int, v float64) float64 { return math.Exp(v) / (1 + math.Exp(v)) }
	xrn, _ := X.Dims()
	prod := mat.NewVecDense(xrn, nil)
	prod.MulVec(X, w)
	prod.AddVec(prod, b)
	h.Apply(sigm, prod)
}
