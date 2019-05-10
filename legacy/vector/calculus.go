package vector

// Add - add v2 to v1 vector
func (v1 *Vector) Add(v2 *Vector) {
	if v1.n != v2.n {
		panic("Must be same size")
	}
	for i := range v1.data {
		v1.data[i] += v2.data[i]
	}
}

//do we need this?
// Sum - return sum of vectors
func Sum(v1, v2 *Vector) (v *Vector) {
	if v1.n != v2.n {
		panic("Must be same size")
	}
	v = new(Vector)
	v.data = make([]float64, v1.n)
	v.n = len(v.data)
	for i := range v1.data {
		v.data[i] = v1.data[i] + v2.data[i]
	}
	return
}

func (v1 *Vector) InnerProd(v2 *Vector) {
	if v1.n != v2.n {
		panic("Must be same size")
	}
	for i := range v1.data {
		v1.data[i] *= v2.data[i]
	}
}

func (v1 *Vector) Mul(s float64) {
	for i := range v1.data {
		v1.data[i] *= s
	}
}
