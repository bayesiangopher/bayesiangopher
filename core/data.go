package core

import (
	"gonum.org/v1/gonum/mat"
)

// Row is row of data
type Row struct {
	Data 		[]float64
	Elements 	int
}

type Train *[]Row

func MakeMatrixFromTrain(train Train) (M *mat.Dense) {
	r := len(*train)
	c := (*train)[0].Elements
	v := make([]float64, r * c)
	for index, row := range *train {
		for idx, element := range row.Data {
			v[c*index + idx] = element
		}
	}
	return mat.NewDense(r, c, v)
}