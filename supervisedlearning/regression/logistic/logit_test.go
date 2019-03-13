package logit

import (
	"io/ioutil"
	"log"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestLogit(t *testing.T) {
	filePath, err := filepath.Abs("iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}

	lines := strings.Split(string(content), "\n")
	lines = lines[:len(lines)-1]
	irisData := make([][]float64, len(lines))
	irisLabels := make([]float64, len(lines))
	for ii, line := range lines {
		vector := strings.Split(line, ",")
		label := vector[len(vector)-1]
		vector = vector[:len(vector)-1]
		floatVector := make([]float64, len(vector))
		for jj := range vector {
			floatVector[jj], err = strconv.ParseFloat(vector[jj], 64)
		}
		irisData[ii] = floatVector
		if label == "Setosa" {
			irisLabels[ii] = 0
			// } else if label == "Versicolor" { //for Multiclass logit comming next patch
			// irisLabels[ii] = 1
		} else {
			irisLabels[ii] = 1
		}
	}

	weights := make([]float64, len(irisData))
	bias := make([]float64, len(irisData))

	weights = fit(irisData, irisLabels, weights, bias, 0.1, 10000)
	result := predict(irisData, weights)
	for i, v := range result {
		if v != irisLabels[i] {
			t.Errorf("Predicted results don't match with input labels")
			break
		}
	}
}
