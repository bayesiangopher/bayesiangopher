package vector

import (
	"fmt"
	"testing"
)

//Needs better structure and more testcases
func TestVec(t *testing.T) {
	a := Zero(2)
	b := One(2)

	fmt.Println(a)
	fmt.Println(b)

	fmt.Println(L1(a, b))
	fmt.Println(L2(a, b))
	fmt.Println(CanberraDistance(a, b))
	a.Add(b)
	b.Mul(1.4)
	fmt.Println(ChebyshevDistance(a, b))

	a.Add(b)
	a.Mul(2)
	b.Mul(4)
	fmt.Println(a)
	fmt.Println(b)
	a.InnerProd(b)
	fmt.Println(a)

	v := Sum(a, b)
	fmt.Println(v)
}
