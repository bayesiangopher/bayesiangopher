package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

const (
	N = 1000000 // Size of vectors
)

func main() {

	u := mat.NewVecDense(N, nil)
	for i := 0; i < N; i++ {
		rand.Seed(int64(i))
		u.SetVec(i, rand.Float64()*100)
	}

	v := mat.NewVecDense(N, nil)
	for i := 0; i < N; i++ {
		rand.Seed(int64(i+N))
		v.SetVec(i, rand.Float64()*100)
	}

	fmt.Printf("Vector 'u': %v\n\n",
		mat.Formatted(u, mat.Prefix(" "), mat.Excerpt(3)))
	fmt.Printf("Vector 'v': %v\n",
		mat.Formatted(v, mat.Prefix(" "), mat.Excerpt(3)))

	// Vector + Vector test:
	w := mat.NewVecDense(N, nil)
	w.AddVec(u, v)
	fmt.Printf("Vector 'w': %v\n\n",
		mat.Formatted(w, mat.Prefix(" "), mat.Excerpt(3)))

	// Add Vector + alpha * Vector test:
	w = mat.NewVecDense(N, nil)
	w.AddScaledVec(u, 2, v)
	fmt.Printf("Vector 'w': %v\n\n",
		mat.Formatted(w, mat.Prefix(" "), mat.Excerpt(3)))

	// Subtract Vector from Vector test:
	w = mat.NewVecDense(N, nil)
	w.SubVec(u, v)
	fmt.Printf("Vector 'w': %v\n\n",
		mat.Formatted(w, mat.Prefix(" "), mat.Excerpt(3)))

	// Scale Vector by alpha test:
	w = mat.NewVecDense(N, nil)
	w.ScaleVec(23, u)
	fmt.Printf("Vector 'w': %v\n\n",
		mat.Formatted(w, mat.Prefix(" "), mat.Excerpt(3)))

	// Dot of Vectors
	d := mat.Dot(u, v)
	fmt.Printf("Vector 'd': %v\n\n", d)

	// Frobenius norm of Vector:
	f := mat.Norm(u, 2)
	fmt.Printf("Frobenius norm of vector 'u': %d", f)

}

//func add_test(u *mat.VecDense, )