package speedtests

// mapVecTest - map of tests names and dims of vectors
var mapVecTest = map[string]int{
	"Size=1024": 1024,
	"Size=16384": 16384,
	"Size=65536": 65536,
	"Size=131072": 131072,
	"Size=262144": 262144,
	"Size=524288": 524288,
}

// mapMatTest - map of tests names and dims of vectors
var mapMatTest = map[string][]int{
	"Size=1024": {32, 32},
	"Size=16384": {128, 128},
	"Size=65536": {256, 256},
	"Size=262144": {512, 512},
	"Size=1048576": {1024, 1024},
}

