package matrix

// Transpose matrix m
func (m *Matrix) Transpose() {
	matrix := make([][]float64, m.m)
	for i := 0; i < m.n; i++ {
		for j := 0; j < m.m; j++ {
			matrix[j] = append(matrix[j], m.data[j][i])
		}
	}
	m.transposed = matrix
}
