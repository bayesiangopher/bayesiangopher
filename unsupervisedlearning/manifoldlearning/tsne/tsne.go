package tsne

import (
	"math"
	"math/rand"
)

const (
	perplexity float64 = 30
	epsilon    float64 = 10
	// NbDims - target dimension.
	NbDims int = 2
)

// Point is a point's coordinates in the target space.
type Point [NbDims]float64

// TSne is the main structure.
type TSne struct {
	iter     int
	length   int
	probas   []float64
	Solution []Point
	gains    []Point
	ystep    []Point
	// Meta gives meta-information about each point, if needed.
	Meta []interface{}
}

// New takes a set of Distancer instances
// and creates matrix P from them using gaussian kernel.
func New(x Distancer, meta []interface{}) *TSne {
	dists := xtod(x)
	length := x.Len()
	tsne := &TSne{
		0,
		length,
		d2p(dists, 30, 1e-4),
		randn2d(length),
		fill2d(length, 1.0),
		make([]Point, length),
		meta,
	}
	return tsne
}

// Step performs a single step of optimization to improve the embedding.
func (tsne *TSne) Step() float64 {
	tsne.iter++
	length := tsne.length
	cost, grad := tsne.costGrad(tsne.Solution)
	var ymean Point
	for i := 0; i < length; i++ {
		for d := 0; d < NbDims; d++ {
			gid := grad[i][d]
			sid := tsne.ystep[i][d]
			gainid := tsne.gains[i][d]
			if sign(gid) == sign(sid) {
				tsne.gains[i][d] = gainid * 0.8
			} else {
				tsne.gains[i][d] = gainid + 0.2
			}
			momval := 0.8
			if tsne.iter < 250 {
				momval = 0.5
			}
			newsid := momval*sid - epsilon*tsne.gains[i][d]*grad[i][d]
			tsne.ystep[i][d] = newsid
			tsne.Solution[i][d] += newsid
			ymean[d] += tsne.Solution[i][d]
		}
	}
	for i := 0; i < length; i++ {
		for d := 0; d < NbDims; d++ {
			tsne.Solution[i][d] -= ymean[d] / float64(length)
		}
	}
	return cost
}

func (tsne *TSne) costGrad(Y []Point) (cost float64, grad []Point) {
	length := tsne.length
	P := tsne.probas
	pmul := 1.0
	if tsne.iter < 100 {
		pmul = 4.0
	}
	Qu := make([]float64, length*length)
	qsum := 0.0
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			dsum := 0.0
			for d := 0; d < NbDims; d++ {
				dhere := Y[i][d] - Y[j][d]
				dsum += dhere * dhere
			}
			qu := 1.0 / (1.0 + dsum)
			Qu[i*length+j] = qu
			Qu[j*length+i] = qu
			qsum += 2 * qu
		}
	}
	Q := make([]float64, length*length)
	for q := range Q {
		Q[q] = math.Max(Qu[q]/qsum, 1e-100)
	}
	cost = 0.0
	grad = make([]Point, length)
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			idx := i*length + j
			cost += -P[idx] * math.Log(Q[idx])
			premult := 4 * (pmul*P[idx] - Q[idx]) * Qu[idx]
			for d := 0; d < NbDims; d++ {
				grad[i][d] += premult * (Y[i][d] - Y[j][d])
			}
		}
	}
	return cost, grad
}

// NormalizeSolution makes all values from the solution in the interval [0; 1].
func (tsne *TSne) NormalizeSolution() {
	var mins [NbDims]float64
	var maxs [NbDims]float64
	for i, pt := range tsne.Solution {
		for j, val := range pt {
			if i == 0 || val < mins[j] {
				mins[j] = val
			}
			if i == 0 || val > maxs[j] {
				maxs[j] = val
			}
		}
	}
	for i, pt := range tsne.Solution {
		for j, val := range pt {
			tsne.Solution[i][j] = (val - mins[j]) / (maxs[j] - mins[j])
		}
	}
}

// Distancer describes a collection of points from which a distance can be computed.
type Distancer interface {
	Len() int
	Distance(i, j int) float64
}

// VectorDistancer - a Distancer implemented for vectors of float64
type VectorDistancer [][]float64

// Len returns the length of the vector.
func (vd VectorDistancer) Len() int { return len(vd) }

// Distance returns the euclidean distance between vd[i] and vd[j].
func (vd VectorDistancer) Distance(i, j int) float64 {
	vi := vd[i]
	vj := vd[j]
	dist := 0.0
	for k, vik := range vi {
		vjk := vj[k]
		dist += (vik - vjk) * (vik - vjk)
	}
	return dist
}

// return 0 mean unit standard deviation random number
func gaussRandom() float64 {
	u := 2*rand.Float64() - 1
	v := 2*rand.Float64() - 1
	r := u*u + v*v
	for r == 0 || r > 1 {
		u = 2*rand.Float64() - 1
		v = 2*rand.Float64() - 1
		r = u*u + v*v
	}
	c := math.Sqrt(-2 * math.Log(r) / r)
	return u * c
}

// return random normal number
func randn(mu, std float64) float64 {
	return mu + gaussRandom()*std
}

// returns 2d array filled with random numbers
func randn2d(n int) []Point {
	res := make([]Point, n)
	for i := range res {
		for j := range res[i] {
			res[i][j] = randn(0.0, 1e-4)
		}
	}
	return res
}

// returns 2d array filled with 'val'
func fill2d(n int, val float64) []Point {
	res := make([]Point, n)
	for i := range res {
		for j := range res[i] {
			res[i][j] = val
		}
	}
	return res
}

// compute pairwise distance in all vectors in X
func xtod(x Distancer) []float64 {
	length := x.Len()
	dists := make([]float64, length*length) // allocate contiguous array
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			d := x.Distance(i, j)
			dists[i*length+j] = d
			dists[j*length+i] = d
		}
	}
	return dists
}

// "constants" for positive and negative infinity
var (
	inf     = math.Inf(1)
	negInf  = math.Inf(-1)
	hTarget = math.Log(perplexity)
)

func d2p(D []float64, perplexity, tol float64) []float64 {
	length := int(math.Sqrt(float64(len(D))))
	pTemp := make([]float64, length*length)
	prow := make([]float64, length)
	for i := 0; i < length; i++ {
		betamin := negInf
		betamax := inf
		beta := 1.0
		const maxtries = 500
		for num := 0; num < maxtries; num++ {
			psum := 0.0
			for j := 0; j < length; j++ {
				if i != j {
					pj := math.Exp(-D[i*length+j] * beta)
					prow[j] = pj
					psum += pj
				} else {
					prow[j] = 0.0
				}
			}
			hHere := 0.0
			for j := 0; j < length; j++ {
				pj := prow[j] / psum
				prow[j] = pj
				if pj > 1e-7 {
					hHere -= pj * math.Log(pj)
				}
			}
			if hHere > hTarget {
				betamin = beta
				if betamax == inf {
					beta = beta * 2
				} else {
					beta = (beta + betamax) / 2
				}
			} else {
				betamax = beta
				if betamin == negInf {
					beta = beta / 2
				} else {
					beta = (beta + betamin) / 2
				}
			}
			if math.Abs(hHere-hTarget) < tol {
				break
			}
		}
		for j := 0; j < length; j++ {
			pTemp[i*length+j] = prow[j]
		}
	}
	probas := make([]float64, length*length)
	length2 := float64(length * 2)
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			probas[i*length+j] = math.Max((pTemp[i*length+j]+pTemp[j*length+i])/length2, 1e-100)
		}
	}
	return probas
}

func sign(x float64) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}
