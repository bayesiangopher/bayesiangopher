package main

import (
	"fmt"
	"testing"
)

func BenchmarkSample(b *testing.B) {
	// Fill the difference:
	//b.N = 1
	//b.N = 500
	//b.N = 10000
	b.N = 1000000
	b.SetBytes(2)
	for i := 0; i < b.N; i++ {
		if x := fmt.Sprintf("%d", 42); x != "42" {
			b.Fatalf("Unexpected string: %s", x)
		}
	}
}