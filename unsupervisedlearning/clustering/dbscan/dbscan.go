package dbscan

import (
	"github.com/bayesiangopher/bayesiangopher/core"
	"gonum.org/v1/gonum/mat"
)

// НУ ОЧЕНЬ НЕОЧЕНЬ РЕАЛИЗОВАЛ, Я ЗНАЮ, Я КАЮСЬ, Я ИСПРАВЛЮСЬ,
// ДА ПРОСТИТ МЕНЯ БОГ ОПЕНСОРСА.
type DBSCAN struct{
	M				*mat.Dense
	minNeighbors 	int
	eps 			float64
	LabeledTrain	[][]mat.VecDense
	Noise			[]mat.VecDense
	visitedPoints 	[]mat.VecDense
	clusteredPoints	[]mat.VecDense
	Labels			[]int
	LabelsCount		int
}

func (dbscan *DBSCAN) Fit(train core.Train, eps float64, minNeighbors int) {
	dbscan.M = core.MakeMatrixFromTrain(train)
	r, _ := dbscan.M.Dims()
	dbscan.minNeighbors = minNeighbors
	dbscan.eps = eps
	for i := 0; i < r; i++ {
		v := mat.NewVecDense(dbscan.M.RowView(i).Len(), nil)
		v.CopyVec(dbscan.M.RowView(i))
		if _, in := checkIn(v, &dbscan.visitedPoints); in { continue }
		dbscan.visitedPoints = append(dbscan.visitedPoints, *v)
		neighbors := region(v, dbscan.M, dbscan.eps)
		if len(*neighbors) < minNeighbors {
			dbscan.Noise = append(dbscan.Noise, *v)
		} else {
			dbscan.Labels = append(dbscan.Labels, len(dbscan.Labels) + 1)
			extendCluster(v, neighbors, dbscan)
		}
	}
}

func region(vec *mat.VecDense, M *mat.Dense, eps float64) *[]mat.VecDense {
	var neighbors []mat.VecDense
	r, _ := M.Dims()
	for i := 0; i < r; i++ {
		vTemp := mat.NewVecDense(M.RowView(0).Len(), nil)
		vTemp.CopyVec(M.RowView(i))
		if dist := core.VecEuclidean(vec, vTemp); dist < eps && dist > 1e-8 {
			neighbors = append(neighbors, *vTemp)
		}
	}
	return &neighbors
}

func extendCluster(v *mat.VecDense, neighbors *[]mat.VecDense, dbscan *DBSCAN) {
	var temp []mat.VecDense
	dbscan.LabeledTrain = append(dbscan.LabeledTrain, temp)
	lastCluster := len(dbscan.LabeledTrain) - 1
	dbscan.clusteredPoints = append(dbscan.clusteredPoints, *v)
	while := len(*neighbors)
	for while > 0 {
		while -= 1
		u := mat.NewVecDense((*neighbors)[len(*neighbors)-1].Len(), nil)
		u.CopyVec(&(*neighbors)[len(*neighbors)-1])
		neighbors = remove(neighbors, len(*neighbors)-1)
		if _, in := checkIn(u, &dbscan.visitedPoints); !in {
			dbscan.visitedPoints = append(dbscan.visitedPoints, *u)
			subNeighbors := region(u, dbscan.M, dbscan.eps)
			if len(*subNeighbors) > dbscan.minNeighbors {
				neighbors = extend(neighbors, subNeighbors)
				while = len(*neighbors)
			}
		}
		if _, in := checkIn(u, &dbscan.clusteredPoints); !in {
			dbscan.clusteredPoints = append(dbscan.clusteredPoints, *u)
			dbscan.LabeledTrain[lastCluster] = append(dbscan.LabeledTrain[lastCluster], *u)
			if i, in := checkIn(u, &dbscan.Noise); in {
				remove(&dbscan.Noise, i)
			}
		}
	}
}

func extend(v, u *[]mat.VecDense) *[]mat.VecDense {
	w := make([]mat.VecDense, len(*v) + len(*u))
	for i := 0; i < len(*v); i++ {
		w[i] = (*v)[i]
	}
	offset := len(*v)
	for i := 0; i < len(*u); i++ {
		w[offset + i] = (*u)[i]
	}
	return &w
}

func remove(checkTarget *[]mat.VecDense, i int) *[]mat.VecDense {
	u := make([]mat.VecDense, len(*checkTarget) - 1)
	for k := 0; k < len(*checkTarget); k++ {
		if k == i { continue }
		if k > i { u[k - 1] = (*checkTarget)[k]; continue }
		u[k] = (*checkTarget)[k]
	}
	return &u
}

func checkIn(v *mat.VecDense, checkTarget *[]mat.VecDense) (int, bool) {
	if checkTarget == nil { return 0, false }
	for i := 0; i < len(*checkTarget); i++ {
		var check int
		for j := 0; j <= v.Len(); j++ {
			if check == v.Len() { return i, true }
			if v.AtVec(j) != (*checkTarget)[i].AtVec(j) || j == v.Len() {
				break
			} else {
				check += 1
			}
		}
	}
	return 0, false
}