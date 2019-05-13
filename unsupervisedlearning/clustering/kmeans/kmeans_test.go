package kmeans

import (
	"fmt"
	"testing"

	"github.com/bayesiangopher/bayesiangopher/core"
)

func TestKmeans(t *testing.T) {
	// func BenchmarkKmeans(b *testing.B) {
	r_train := core.CSVReader{Path: "train.csv"}
	r_test := core.CSVReader{Path: "test.csv"}
	// r := core.CSVReader{Path: "../../../datasets/the_iris_kmeans_dataset.csv"}
	train := r_train.Read(true)
	test := r_test.Read(true)

	k := 3
	kmns := KMNS{state_init: false}
	kmns.Fit(train, k)
	fmt.Println(kmns.labels)
	lbs := kmns.Predict(test)
	fmt.Println(lbs)
}
