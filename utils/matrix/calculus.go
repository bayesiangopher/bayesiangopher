package matrix

import (
	"errors"
)

// IsEqual do checking of equality.
func (m1 Matrix) IsEqual(m2 Matrix) (bool, error) {
	if m1.n != m2.n || m1.m != m2.m {
		return false, errors.New("Matrixes must be same size.")
	}
	for i := range m1.data {
		for j := range m1.data[i] {
			if m1.data[i][j] != m2.data[i][j] {
				return false, nil
			}
		}
	}
	return true, nil
}

// Dot do multiplication of matrixes
func (m1 Matrix) Dot(m2 Matrix) (*Matrix, error) {
	if m1.m != m2.n {
		return nil, errors.New("Can't do matrix multiplication.")
	}
	M := new(Matrix)
	matrix := make([][]float64, m1.n)
	for i := 0; i < m1.n; i++ {
		for j := 0; j < m2.n; j++ {
			if len(matrix[i]) < 1 {
				matrix[i] = make([]float64, m2.m)
			}
			matrix[i][j] += m1.data[i][j] * m2.data[j][i]
		}
	}
	M.data = matrix
	return M, nil
}
