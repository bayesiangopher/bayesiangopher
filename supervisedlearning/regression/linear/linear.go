// Simple Linear Regressions

package linear

import (
	"bufio"
	"flag"
	"fmt"
	"image/color"
	"log"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

var iterations int

func main() {
	flag.IntVar(&iterations, "n", 1000, "number of iterations")
	flag.Parse()

	xys, err := readData("data.txt")
	if err != nil {
		log.Fatalf("could not read data.txt: %v", err)
	}
	_ = xys
	err = plotData("out.png", xys)
	if err != nil {
		log.Fatalf("couldn't plot data: %v", err)
	}
}

func readData(path string) (plotter.XYs, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var xys plotter.XYs
	s := bufio.NewScanner(f)
	for s.Scan() {
		var x, y float64
		_, err := fmt.Sscanf(s.Text(), "%f,%f", &x, &y)
		if err != nil {
			log.Printf("discarding bad data point %q: %v", s.Text(), err)
		}
		xys = append(xys, struct{ X, Y float64 }{x, y})
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("could no scan: %v", err)
	}
	return xys, nil
}

func plotData(path string, xys plotter.XYs) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("couldn't creare %s: %v", path, err)
	}

	p, err := plot.New()
	if err != nil {
		return fmt.Errorf("couldn't create plot: %v", err)
	}

	s, err := plotter.NewScatter(plotter.XYs(xys))
	if err != nil {
		return fmt.Errorf("couldn't creaete scatter: %v", err)
	}
	s.GlyphStyle.Shape = draw.CrossGlyph{}
	s.Color = color.RGBA{R: 255, A: 255}
	p.Add(s)

	m, c := linearRegression(xys, 0.01)

	l, err := plotter.NewLine(plotter.XYs{
		{3, 3*m + c}, {20, 20*m + c},
	})
	if err != nil {
		return fmt.Errorf("couldn't create line: %v", err)
	}
	p.Add(l)

	wt, err := p.WriterTo(256, 256, "png")
	if err != nil {
		return fmt.Errorf("couldn't create writer: %v", err)
	}
	_, err = wt.WriteTo(f)
	if err != nil {
		return fmt.Errorf("couldn't write to %s: %v", path, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("couldn't close %s: %v", path, err)
	}
	return nil
}

func linearRegression(xys plotter.XYs, alpha float64) (m, c float64) {
	for i := 0; i < iterations; i++ {
		dm, dc := computeGradient(xys, m, c)
		m += -dm * alpha
		c += -dc * alpha
	}

	fmt.Printf("cost(%.2f, %.2f) = %.2f\n", m, c, computeCost(xys, m, c))

	return m, c
}

func computeCost(xys plotter.XYs, m, c float64) float64 {
	// cost = 1/N * sum((y - (m*x + c))^2)
	s := 0.0
	for _, xy := range xys {
		d := xy.Y - (xy.X*m + c)
		s += d * d
	}
	return s / float64(len(xys))
}

func computeGradient(xys plotter.XYs, m, c float64) (dm, dc float64) {
	// cost = 1/N * sum((y - (m*x + c))^2)
	// cost/dm = 2/N * sum((y - (m*x + c)) * (-x)
	// cost/dc = 2/N * sum((y - (m*x + c)) * (-1)
	for _, xy := range xys {
		d := xy.Y - (xy.X*m + c)
		dm += -xy.X * d
		dc += -d
	}
	n := float64(len(xys))
	return 2 / n * dm, 2 / n * dc
}
