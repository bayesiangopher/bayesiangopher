package core

import (
	"fmt"
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

// MatPrint print Matrix to Stdout
func MatPrint(A mat.Matrix) {
	fmt.Printf("%v\n", mat.Formatted(A, mat.Prefix(" "), mat.Excerpt(3)))
}

// VecPrint print vector in stdout
func VecPrint(v *mat.VecDense) {
	fmt.Printf("%v\n",
		mat.Formatted(v, mat.Prefix(" "), mat.Excerpt(3)))
}