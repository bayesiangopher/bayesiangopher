package logit

import (
	"testing"

	"github.com/bayesiangopher/bayesiangopher/core"
)

//Speed of fitting model for BreastCancer dataset using SGD solver
func BenchmarkFitCanserSGD(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/breast_canser.csv"}
	train := r.Read(true)

	lgt := LGR{solver: SGD, batchs: 20, lrate: 0.0001}
	lgt.Fit(train, 0)
}

//Speed of fitting model for BreastCancer dataset using GD solver
func BenchmarkFitCanserGD(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/breast_canser.csv"}
	train := r.Read(true)

	lgt := LGR{solver: GD, lrate: 0.0001}
	lgt.Fit(train, 0)
}

//Speed of fitting model for Iris dataset using SGD solver
func BenchmarkFitIrisSGD(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/iris.csv"}
	train := r.Read(true)

	lgt := LGR{solver: GD, lrate: 0.0001}
	lgt.Fit(train, 4)
}

//Speed of fitting model for Iris dataset using GD solver
func BenchmarkFitIrisGD(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/iris.csv"}
	train := r.Read(true)

	lgt := LGR{solver: GD, lrate: 0.0001}
	lgt.Fit(train, 4)
}
