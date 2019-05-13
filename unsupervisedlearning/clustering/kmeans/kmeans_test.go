package kmeans

import (
	"testing"

	"github.com/bayesiangopher/bayesiangopher/core"
)

//TODO: Plot results for detailed check
func TestKmeans(t *testing.T) {
	r := core.CSVReader{Path: "../../../datasets/kmeans_iris.csv"}
	train := r.Read(true)

	k := 3
	kmns := KMNS{}
	//Fit (data, k)
	kmns.Fit(train, k)
	// filePath, err := filepath.Abs("iris.csv")
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// content, err := ioutil.ReadFile(filePath)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// lines := strings.Split(string(content), "\n")
	// lines = lines[:len(lines)-1]
	// irisData := make([][]float64, len(lines))
	// irisLabels := make([]string, len(lines))
	// for ii, line := range lines {
	// 	vector := strings.Split(line, ",")
	// 	label := vector[len(vector)-1]
	// 	vector = vector[:len(vector)-1]
	// 	floatVector := make([]float64, len(vector))
	// 	for jj := range vector {
	// 		floatVector[jj], err = strconv.ParseFloat(vector[jj], 64)
	// 	}
	// 	irisData[ii] = floatVector
	// 	irisLabels[ii] = label
	// }
	// labels, err := Kmeans(irisData, 3, L1, 10)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Println(labels)
}
