package core

import "gonum.org/v1/gonum/mat"

// Row is row of data
type Row struct {
	data 		[]float64
	elements 	int
}

func MakeMatrixFromTrain(train *[]Row) (M *mat.Dense) {
	r := len(*train)
	c := (*train)[0].elements
	v := make([]float64, r * c)
	for _, row := range *train {
		for _, element := range row.data {
			v = append(v, element)
		}
	}
	return mat.NewDense(r, c, v)
}