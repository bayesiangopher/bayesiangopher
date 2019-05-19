package kmeans

import (
	"errors"
	"fmt"
	"log"
	"testing"

	"github.com/bayesiangopher/bayesiangopher/core"
)

//Speed of fitting model for Xclara dataset
func BenchmarkXclaraFitKmeans(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_xclara_cluster_2.5k_dataset.csv"}
	train := r.Read(true)
	k := 3
	kmns := KMNS{}
	b.ReportAllocs()
	b.N = 10
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StartTimer()
		err := kmns.Fit(train, k)
		if err != nil { log.Fatal(errors.New("ошибка фита")) }
		b.StopTimer()
	}
}

//Speed of fitting model for Iris dataset
func BenchmarkIrisFitKmeans(b *testing.B) {
	r := core.CSVReader{Path: "../../../datasets/the_train_randblobs_cluster.csv"}
	train := r.Read(true)

	k := 3
	kmns := KMNS{}
	err := kmns.Fit(train, k)
	if err != nil { log.Fatal(errors.New("ошибка фита")) }
}

//Speed of fitting model and predicting labels for randomly generated blobs
func BenchmarkRandBlobsFitPredictKmeans(b *testing.B) {
	r_train := core.CSVReader{Path: "../../../datasets/the_train_randblobs_cluster.csv"}
	r_test := core.CSVReader{Path: "../../../datasets/the_test_randblobs_cluster.csv"}
	train := r_train.Read(true)
	test := r_test.Read(true)

	k := 3
	kmns := KMNS{}
	err := kmns.Fit(train, k)
	if err != nil { log.Fatal(errors.New("ошибка фита")) }
	fmt.Println(kmns.labels)
	lbs := kmns.Predict(test)
	fmt.Println(lbs)
}

//Fitting model and predicting labels for randomly generated blobs
func TestRandBlobsFitPredictKmeans(t *testing.T) {
	r_train := core.CSVReader{Path: "../../../datasets/the_train_randblobs_cluster.csv"}
	r_test := core.CSVReader{Path: "../../../datasets/the_test_randblobs_cluster.csv"}
	train := r_train.Read(true)
	test := r_test.Read(true)

	k := 3
	kmns := KMNS{}
	err := kmns.Fit(train, k)
	if err != nil { log.Fatal(errors.New("ошибка фита")) }
	fmt.Println(kmns.labels)
	lbs := kmns.Predict(test)
	fmt.Println(lbs)
}
