package matrix

// IsEqual do checking of equality.
func (m1 Matrix) IsEqual(m2 Matrix) bool {
	if m1.n != m2.n || m1.m != m2.m {
		panic("Dimensions of matrixes must be the equal!")
	}
	for i := range m1.data {
		for j := range m1.data[i] {
			if m1.data[i][j] != m2.data[i][j] {
				return false
			}
		}
	}
	return true
}
