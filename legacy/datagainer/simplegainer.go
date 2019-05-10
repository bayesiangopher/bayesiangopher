package datagainer

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

type DataFormat struct {
	Circle     bool
	Rectangle  bool
	Line       bool
	Xlim, Ylim float64
	DataDim    int
	Paint      bool
	Randseed   float64
}

func SetDataFormat(figures [3]bool, xlim, ylim float64, dataDim int,
	paint bool, randseed float64) *DataFormat {
	Data := DataFormat{}
	Data.Circle = figures[0]
	Data.Rectangle = figures[1]
	Data.Line = figures[2]
	Data.Xlim = xlim
	Data.Ylim = ylim
	Data.DataDim = dataDim
	Data.Paint = paint
	Data.Randseed = randseed
	return &Data
}

func Circle(xlim float64) [2]float64 {
	x := rand.Float64()*xlim/2 - xlim/4
	y := math.Sqrt(math.Pow(xlim/4, 2) - math.Pow(x, 2))
	return [2]float64{x, y}
}

func Rectangle(xlim, ylim float64) [2]float64 {
	x := rand.Float64()*xlim - xlim/2
	y := ylim / 4
	return [2]float64{x, y}
}

func Line(xlim, ylim float64) [2]float64 {
	x := rand.Float64() * xlim / 2
	y := x
	return [2]float64{x, y}
}

func MakeData(data *DataFormat) {
	DataSet := make([][]float64, data.DataDim-1)
	for i := 0; i < data.DataDim-1; i++ {
		DataSet[i] = make([]float64, 2)
	}
	if data.Circle {
		for i := 0; i < data.DataDim/3; i++ {
			for j := 0; j < 2; j++ {
				unit := Circle(data.Xlim)
				DataSet[i][0] = unit[0]
				DataSet[i][1] = unit[1]
			}
		}
	}
	if data.Rectangle {
		for i := data.DataDim / 3; i < data.DataDim/3*2; i++ {
			for j := 0; j < 2; j++ {
				unit := Rectangle(data.Xlim, data.Ylim)
				DataSet[i][0] = unit[0]
				DataSet[i][1] = unit[1]
			}
		}
	}
	if data.Line {
		for i := data.DataDim / 3 * 2; i < data.DataDim/3*3; i++ {
			for j := 0; j < 2; j++ {
				unit := Line(data.Xlim, data.Ylim)
				DataSet[i][0] = unit[0]
				DataSet[i][1] = unit[1]
			}
		}
	}
	file, err := os.Create("testdata.csv")
	if err != nil {
		fmt.Println("Unable to create file:", err)
		os.Exit(1)
	}
	defer file.Close()
	for i := 0; i < len(DataSet); i++ {
		file.WriteString(strconv.FormatFloat(DataSet[i][0], 'f', -1, 64))
		file.WriteString(",")
		file.WriteString(strconv.FormatFloat(DataSet[i][1], 'f', -1, 64))
		file.WriteString("\n")
	}
}

func main() {
	figure := [3]bool{true, true, true}
	data := SetDataFormat(figure, 6, 6, 100, false, 1)
	MakeData(data)
}
