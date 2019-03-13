package logit

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
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

	x := make([][]float64, 3)
	w := make([]float64, 3)
	b := make([]float64, 3)

	for i := 0; i < 3; i++ {
		v := make([]float64, 3)
		for j := 0; j < 3; j++ {
			v[j] = float64(i + j + rand.Intn(3))
		}
		x[i] = v
		w[i] = float64(i + 1 + rand.Intn(5))
		b[i] = 0
	}

	fmt.Println(x)
	fmt.Println(w)

	fmt.Println(dot(x, w))

	fmt.Println(sigm(x, w, b))
	fmt.Println(loss(sigm(x, w, b), w))
	fmt.Println(grad_desc(x, sigm(x, w, b), w))

	fmt.Println(update_weight_loss(w, grad_desc(x, sigm(x, w, b), w), 0.1))

	weights := make([]float64, len(irisData))
	bias := make([]float64, len(irisData))

	weights = fit(irisData, irisLabels, weights, bias, 0.1, 10000)
	fmt.Println(weights)
	fmt.Println(predict(irisData, weights))
}
