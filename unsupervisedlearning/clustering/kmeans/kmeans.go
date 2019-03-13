package kmeans

import (
	"math"
	"math/rand"
)

//Rewrite in terms of utils/vector
type Vec []float64

type ClusterObj struct {
	ClusterNum int
	Vec
}

type DistFunc func(vec1, vec2 []float64) (float64, error)

func L1(vec1, vec2 []float64) (float64, error) {
	var dist float64
	for i := range vec1 {
		dist += math.Abs(vec1[i] - vec2[i])
	}
	return dist, nil
}

func L2(vec1, vec2 []float64) (float64, error) {
	var dist float64
	for i := range vec1 {
		dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i])
	}
	return math.Sqrt(dist), nil
}

func CanberraDistance(vec1, vec2 []float64) (float64, error) {
	var dist float64
	for i := range vec1 {
		dist += (math.Abs(vec1[i]-vec2[i]) / (math.Abs(vec1[i]) + math.Abs(vec2[i])))
	}
	return dist, nil
}

func (vec Vec) Add(vec1 Vec) {
	for i, val := range vec1 {
		vec[i] += val
	}
}

func (vec Vec) Mul(scal float64) {
	for i := range vec {
		vec[i] *= scal
	}
}

func near(obj ClusterObj, centers []Vec, distFunc DistFunc) (int, float64) {
	cluster_idx := 0
	minDist, _ := distFunc(obj.Vec, centers[0])
	for i := 1; i < len(centers); i++ {
		dist, _ := distFunc(obj.Vec, centers[i])
		if dist < minDist {
			minDist = dist
			cluster_idx = i
		}
	}

	return cluster_idx, minDist
}

func seed(data []ClusterObj, k int, distFunc DistFunc) []Vec {
	s := make([]Vec, k)
	// rand.Seed(time.Now().UnixNano())
	s[0] = data[rand.Intn(len(data))].Vec
	d2 := make([]float64, len(data))
	for i := 1; i < k; i++ {
		var sum float64
		for j, obj := range data {
			_, d2[j] = near(obj, s[:i], distFunc)
			sum += d2[j]
		}
		target := rand.Float64() * sum
		var j int
		for sum = d2[0]; sum < target; sum += d2[j] {
			j++
		}
		s[i] = data[j].Vec
	}
	return s
}

func kmeans(data []ClusterObj, centers []Vec, distFunc DistFunc, threshold int) ([]ClusterObj, error) {
	var counter int
	for i, obj := range data {
		nearestCluster, _ := near(obj, centers, distFunc)
		data[i].ClusterNum = nearestCluster
	}
	cLen := make([]int, len(centers))
	for n := len(data[0].Vec); ; {
		for i := range centers {
			centers[i] = make(Vec, n)
			cLen[i] = 0
		}
		for _, obj := range data {
			centers[obj.ClusterNum].Add(obj.Vec)
			cLen[obj.ClusterNum]++
		}
		for i := range centers {
			centers[i].Mul(1 / float64(cLen[i]))
		}
		var changed int
		for i, obj := range data {
			if nearestCluster, _ := near(obj, centers, distFunc); nearestCluster != obj.ClusterNum {
				changed++
				data[i].ClusterNum = nearestCluster
			}
		}
		counter++
		if counter == 0 || counter > threshold {
			return data, nil
		}
	}
}

func Kmeans(rawData [][]float64, k int, distFunc DistFunc, threshold int) ([]int, error) {
	data := make([]ClusterObj, len(rawData))
	for i, v := range rawData {
		data[i].Vec = v
	}

	seeds := seed(data, k, distFunc)
	clustData, err := kmeans(data, seeds, distFunc, threshold)
	lables := make([]int, len(clustData))
	for i, obj := range clustData {
		lables[i] = obj.ClusterNum
	}
	return lables, err
}