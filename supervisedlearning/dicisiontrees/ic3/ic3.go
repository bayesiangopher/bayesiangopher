package ic3

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

type gain map[string]float64
type dataset struct {
	head      []string
	rows      [][]string
	rowTotal  float64
	counts    map[string]map[string]map[string]float64
	entropies map[string]float64
	gains     gain
}

// ID3 api algorithm
func ID3(path string, depth int) (subdataset *dataset) {
	dataset := set(path)
	return id3(dataset, 0)
}

func id3(dataset *dataset, depth int) (subdataset *dataset) {
	dataset.calculateCounts()
	if dataset.isHaveOneResult() {
		fmt.Print("|", strings.Repeat("	", depth*3), "|->> ")
		_, value := dataset.resultClass()
		fmt.Println(value)
	}
	dataset.calculateEntropies()
	dataset.calculateGains()
	for value, _ := range dataset.counts[dataset.gains.max()] {
		fmt.Print("|", strings.Repeat("	", depth*3), "|-")
		fmt.Println(dataset.gains.max(), " => ", value)
		id3(dataset.subDataset(value), depth+1)
	}
	return
}

// return elements of n column in dataset.rows
func (d *dataset) columns(n int) (col []string) {
	col = make([]string, 0)
	for _, row := range d.rows {
		col = append(col, row[n])
	}
	return
}

// return id of named column based of dateset.head
// return -1 if there's no such head
func (d *dataset) id(name string) (id int) {
	for k, v := range d.head {
		if v == name {
			return k
		}
	}
	return -1
}

// return ID of last column of dataset
func (d *dataset) classID() (id int) {
	return (len(d.head) - 1)
}

// return name of last column of dataset
func (d *dataset) className() (name string) {
	return d.head[d.classID()]
}

// return value of id element in last column of dataset
func (d *dataset) classRow(id int) (class string) {
	return d.rows[id][d.classID()]
}

// return last column and value of first row in dataset
func (d *dataset) resultClass() (column, value string) {
	return d.className(), d.classRow(0)
}

func (d *dataset) entropy(count, sum float64) (v float64) {
	p := count / sum
	v = p * float64(math.Log2(float64(p)))
	return
}

func (d *dataset) isHaveOneResult() bool {
	return len(d.counts[d.className()]) < 2
}

func (d *dataset) calculateCounts() {
	d.rowTotal = float64(len(d.rows))
	d.counts = make(map[string]map[string]map[string]float64)
	for n, v := range d.head {
		d.counts[v] = make(map[string]map[string]float64)
		for id, row := range d.columns(n) {
			if d.counts[v][row] == nil {
				d.counts[v][row] = make(map[string]float64)
			}
			d.counts[v][row]["Total"]++
			d.counts[v][row][d.classRow(id)]++
		}
	}
}

func (d *dataset) calculateEntropies() {
	d.entropies = make(map[string]float64)

	for _, count := range d.counts[d.className()] {
		d.entropies["Class"] -= d.entropy(float64(count["Total"]), d.rowTotal)
	}

	for _, row := range d.counts {
		for _, cell := range row {
			for k, v := range cell {
				if k == "Total" || k == "Entropy" {
					continue
				}
				cell["Entropy"] -= d.entropy(v, cell["Total"])
			}
		}
	}

	for k, v := range d.counts {
		if k == d.className() {
			continue
		}
		for _, cell := range v {
			d.entropies[k] += cell["Total"] / d.rowTotal * cell["Entropy"]
		}
	}
}

func (d *dataset) calculateGains() {
	d.gains = make(map[string]float64)
	for k, entropy := range d.entropies {
		if k == "Class" {
			continue
		}
		d.gains[k] = d.entropies["Class"] - entropy
	}
}

func (d *dataset) subDataset(value string) (subDataset *dataset) {
	subDataset = new(dataset)
	subDataset.head = d.head
	maxGainKey := d.id(d.gains.max())

	for _, row := range d.rows {
		if row[maxGainKey] == value {
			subDataset.rows = append(subDataset.rows, row)
		}
	}
	return
}

func (g gain) max() string {
	maxValue := 0.0
	maxKey := ""
	for k, v := range g {
		if v > maxValue {
			maxKey = k
			maxValue = v
		}
	}
	return maxKey
}

func set(path string) (data *dataset) {
	data = new(dataset)

	file, err := os.Open(path)
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	data.head = rows[0]
	data.rows = rows[1:]
	return
}
