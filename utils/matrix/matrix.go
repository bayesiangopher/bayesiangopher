package matrix

// Matrix - general struct for matrixes.
type Matrix struct {
	n             int
	m             int
	data          [][]float64
	determinator  float64
	orthogonality bool
	transposed    [][]float64
}
