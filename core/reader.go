package core

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

type Reader interface {
	Read()
}

type CSVReader struct {
	Path string
	Rows int
	Data []Row
}

// Read read data from csv file to CSVReader instance
func (r *CSVReader) Read(makeTrain bool) (train Train) {
	source, _ := os.Open(r.Path)
	defer source.Close()
	for row := range rowsProcessing(source) {
		r.Data = append(r.Data, Row{Data: row, Elements: len(row)})
	}
	if makeTrain { return &r.Data } else { return nil }
}

func rowsProcessing(f io.Reader) (ch chan []float64) {
	ch = make(chan []float64, 32)
	go func() {
		r := csv.NewReader(f)
		if _, err := r.Read(); err != nil { log.Fatal(err) }
		defer close(ch)
		for {
			row, err := r.Read()
			if err == io.EOF { break }
			if err != nil { log.Fatal(err) }
			var floatRow []float64
			for _, el := range row {
				temp, err := strconv.ParseFloat(el, 64)
				if err != nil { log.Fatal(err) }
				floatRow = append(floatRow, temp)
			}
			ch <- floatRow
		}
	}()
	return
}