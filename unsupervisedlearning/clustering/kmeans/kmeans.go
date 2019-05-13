package kmeans

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/bayesiangopher/bayesiangopher/core"
)

type KMNS struct {
	max_iter   int
	tol        float64
	centroids  *mat.Dense
	labels     []int //*mat.VecDense
	state_init bool  //Random (false) or determined (true) state for centroid initialization
}

func (km *KMNS) Fit(train core.Train, k int) (err error) {
	//Prepare data:
	x_train := core.MakeMatrixFromTrain(train)
	//Set optional parameters
	if km.max_iter == 0 {
		km.max_iter = 300
	}
	if !km.state_init {
		rand.Seed(time.Now().UnixNano())
	}
	//Getting the best
	iter := 0
	n_instance, n_features := x_train.Dims()
	//init centroids
	var ctrs []float64
	for i := 0; i < k; i++ {
		r_inst := rand.Intn(n_instance)
		for m := 0; m < n_features; m++ {
			ctrs = append(ctrs, x_train.At(r_inst, m))
		}
	}
	km.centroids = mat.NewDense(k, n_features, ctrs)
	prev_ctrs := mat.NewDense(k, n_features, nil)
	norm := normDense(km.centroids, prev_ctrs)

	for norm != km.tol {
		if iter++; iter > km.max_iter {
			break
		}
		prev_ctrs.Copy(km.centroids)
		//assign all points in ds to centroids
		km.labels = make([]int, n_instance)
		for i := 0; i < n_instance; i++ {
			dist := make([]float64, k)
			for j := 0; j < k; j++ {
				dist[j] = core.VecEuclidean(mat.VecDenseCopyOf(km.centroids.RowView(j)), mat.VecDenseCopyOf(x_train.RowView(i)))
			}
			km.labels[i] = minElemIdx(dist)
		}
		// for all clusters upd centroids
		for i := 0; i < k; i++ {
			for j := 0; j < n_features; j++ {
				val := 0.0
				pn := 0
				for idx, elem := range km.labels {
					if elem == i {
						val += x_train.At(idx, j)
						pn += 1
					}
				}
				km.centroids.Set(i, j, val/float64(pn))
			}
		}
		norm = normDense(km.centroids, prev_ctrs)
	}
	return nil
}

func (km *KMNS) Predict(test core.Train) (labels []int) {
	x_test := core.MakeMatrixFromTrain(test)
	n_instance, _ := x_test.Dims()
	k, _ := km.centroids.Dims()
	for i := 0; i < n_instance; i++ {
		dist := make([]float64, k)
		for j := 0; j < k; j++ {
			dist[j] = core.VecEuclidean(mat.VecDenseCopyOf(km.centroids.RowView(j)), mat.VecDenseCopyOf(x_test.RowView(i)))
		}
		labels = append(labels, minElemIdx(dist))
	}
	return labels
}

func normDense(a, b *mat.Dense) float64 {
	rn, cn := a.Dims()
	c := mat.NewDense(rn, cn, nil)
	c.Sub(b, a)
	return mat.Norm(c, 2)
}

func minElemIdx(d []float64) (idx int) {
	min := d[0]
	for i := 0; i < len(d); i++ {
		if d[i] < min {
			min = d[i]
			idx = i
		}
	}
	return idx
}
