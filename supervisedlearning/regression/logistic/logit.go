package logit

import (
	"math"
)

//put to utils/matrix and use {}interfaces for dot product like:
// mat*vec; vec.T*mat; vec.T*vec.
//Also add dimension check
//Or just base all mat operations on gonum
//Dot product NxM * Mx1
func dot(a [][]float64, b []float64) []float64 {
	c := make([]float64, len(b))
	for i, v := range a {
		var sum float64
		for j := range v {
			sum += b[j] * a[i][j]
		}
		c[i] = sum
	}
	return c
}

//also as matrix method
func exp(a []float64) []float64 {
	b := make([]float64, len(a))
	for i, v := range a {
		b[i] = math.Exp(v)
	}
	return b
}

func logn(a []float64) []float64 {
	b := make([]float64, len(a))
	for i, v := range a {
		b[i] = math.Log(v)
	}
	return b
}

func mul(a []float64, s float64) []float64 {
	b := make([]float64, len(a))
	for i, v := range a {
		b[i] = v * s
	}
	return b
}

func plus(a []float64, s float64) []float64 {
	b := make([]float64, len(a))
	for i, v := range a {
		b[i] = v + s
	}
	return b
}

//RETURN EVERY WHERE
//interface here
func plusvec(a []float64, b []float64) []float64 {
	c := make([]float64, len(a))
	for i, v := range a {
		c[i] = v + b[i]
	}
	return c
}

func divide(a []float64, s float64) []float64 {
	b := make([]float64, len(a))
	for i, v := range a {
		b[i] = s / v
	}
	return b
}

func mulvec(a, b []float64) []float64 {
	c := make([]float64, len(a))
	for i, v := range a {
		c[i] = v * b[i]
	}
	return c
}

func transpose(a [][]float64) [][]float64 {
	b := make([][]float64, len(a[0]))
	for i := range b {
		b[i] = make([]float64, len(a))
	}
	for i, v := range a {
		for j, r := range v {
			b[j][i] = r
		}
	}
	return b
}

// func transpose(a [][]float64) [][]float64 {
// 	b := make([][]float64, len(a[0]))
// 	for i, v := range a {
// 		vec := make([]float64, len(v))
// 		for j := range v {
// 			b
// 		}
// 	}
// }

func sigm(x [][]float64, w, b []float64) []float64 {
	z := dot(x, w)
	//all this crap on C?
	return divide(plus(exp(mul(plusvec(z, b), -1)), 1), 1)
}

func mean(a []float64) float64 {
	var m float64
	for _, v := range a {
		m += v
	}
	return m
}

func loss(h, y []float64) float64 {
	return mean(plusvec(mulvec(mul(y, -1), logn(h)), mulvec(plus(y, 1), logn(plus(mul(h, -1), 1)))))
}

//grad_asce next commit
func grad_desc(x [][]float64, h, y []float64) []float64 {
	return dot(transpose(x), mul(plusvec(h, mul(y, -1)), 1/float64(len(y))))
}

func update_weight_loss(w, grad []float64, lr float64) []float64 {
	return plusvec(w, mul(grad, -lr))
}

func fit(x [][]float64, y, w, b []float64, lr float64, iter int) []float64 {
	for i := 0; i < iter; i++ {
		h := sigm(x, w, b)
		grad := grad_desc(x, h, y)
		w = update_weight_loss(w, grad, lr)
	}
	return w
}

func predict(x [][]float64, w []float64) []float64 {
	b := make([]float64, len(w))
	w = sigm(x, w, b)
	for i, v := range w {
		if v >= 0.5 {
			b[i] = 1
		} else {
			b[i] = 0
		}
	}
	return b
}
