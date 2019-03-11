// https://nabr.com/post/322034/ - base information

package dbscan

const (
	NOISE		= false
	CLUSTERED	= true
)

type Clusterable interface {
	Distance(c interface{}) float64
	GetID() string
}

type Cluster []Clusterable

func Clusterize(objects []Clusterable, minPts int, eps float64) []Cluster {
	clusters	:= make([]Cluster, 0)
	visited 	:= map[string]bool{}
	for _, point := range objects {
		neighbours := findNeighbours(point, objects, eps)
		if len(neighbours) + 1 >= minPts {
			visited[point.GetID()] = CLUSTERED
			cluster := make(Cluster, 1)
			cluster[0] = point
			cluster = expandCluster(cluster, neighbours, visited, minPts, eps)
			if len(cluster) >= minPts {
				cluster = append(clusters, cluster)
			}
		} else {
			visited[point.GetID()] = NOISE
		}
	}
	return clusters
}

func findNeighbours(point Clusterable, points []Clusterable, eps float64) []Clusterable {
	neighbours := make([]Clusterable, 0)
	for _, potNeigb := range points {
		if point.GetID() != potNeigb.GetID() && potNeigb.Distance(point) <= eps {
			neighbours = append(neighbours, potNeigb)
		}
	}
	return neighbours
}

func expandCluster(cluster Cluster, neighbours []Clusterable, 
			visited map[string]bool, minPts int, eps float64) Cluster {
	seed := make([]Clusterable, len(neighbours))
	copy(seed, neighbours)
	for _, point := range seed {
		pointState, isVisited := visited[point.GetID()]
		if !isVisited {
			currentNeighbours := findNeighbours(point, seed, eps)
			if len(currentNeighbours)+1 >= minPts {
				visited[point.GetID()] = CLUSTERED
				cluster = merge(cluster, currentNeighbours)
			}
		}

		if isVisited && pointState == NOISE {
			visited[point.GetID()] = CLUSTERED
			cluster = append(cluster, point)
		}
	}

	return cluster
}

func merge(one []Clusterable, two []Clusterable) []Clusterable {
	mergeMap := make(map[string]Clusterable)
	putAll(mergeMap, one)
	putAll(mergeMap, two)
	merged := make([]Clusterable, 0)
	for _, val := range mergeMap {
		merged = append(merged, val)
	}

	return merged
}

func putAll(m map[string]Clusterable, list []Clusterable) {
	for _, val := range list {
		m[val.GetID()] = val
	}
}