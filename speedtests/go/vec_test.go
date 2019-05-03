package speedtests

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

// TestCreateRandomVector testing speed and result of work
// createRandomVector function
func TestCreateRandomVector(t *testing.T) {
	for key, value := range mapVecTest {
		t.Run(key, testCreateRandomVectorFunc(value))
	}
}

func testCreateRandomVectorFunc(N int) func(t *testing.T) {
	return func(t *testing.T) {
		v := createRandomVector(N)
		fmt.Printf("Created vector 'v' with dimension: %d \n", v.Len()); vecPrint(v)
	}
}

// TestScaleVector testing correctness of
// scaleVector (gonum: ScaleVec) function
func TestScaleVector(t *testing.T) {
	v := createRandomVector(1024)
	check := v.AtVec(5)
	fmt.Printf("Created vector 'v' with dimension: %d\n", v.Len()); vecPrint(v)
	scaleVector(v, 5.25)
	fmt.Println("Vector 'v' after scaling:"); vecPrint(v)
	if v.AtVec(5) / check != 5.25 {
		t.Fatal("Ошибка масштабирования.")
	}
}

// TestFrobeniusNormOfVector testing correctness of
// frobeniusNormOfVector (gonum: Norm(v, 2) function
func TestFrobeniusNormOfVector(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	v := mat.NewVecDense(len(data), data)
	fmt.Printf("Created vector 'v' with dimension: %d\n", v.Len()); vecPrint(v)
	f := frobeniusNormOfVector(v)
	if f != 19.621416870348583  {
		t.Fatal("Неправильно найдена норма Фробениуса.")
	}
	fmt.Printf("Frobenius norm of vector 'v': %v \n", f)
}

// TestAdditionOfVectors testing correctness of
// additionOfVectors (gonum: AddVec) function
func TestAdditionOfVectors(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	dataTest := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
	v := mat.NewVecDense(len(data), data)
	u := mat.NewVecDense(len(data), data)
	resultTest := mat.NewVecDense(len(dataTest), dataTest)
	w := additionOfVectors(u, v, 0.0)
	if froTest := frobeniusNormOfVector(w); froTest != frobeniusNormOfVector(resultTest) {
		t.Fatal("Векторы сложились неправильно.")
	}
}

// TestSubtractOfVectors testing correctness of
// subtractOfVectors (gonum: SubVec) function
func TestSubtractOfVectors(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	dataTest := []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	v := mat.NewVecDense(len(data), data)
	u := mat.NewVecDense(len(data), data)
	resultTest := mat.NewVecDense(len(dataTest), dataTest)
	w := subtractOfVectors(u, v)
	if froTest := frobeniusNormOfVector(w); froTest != frobeniusNormOfVector(resultTest) {
		t.Fatal("Вычитание выполнено с ошибкой.")
	}
}

// TestDotOfVectors testing correctness of
// dotOfVectors (gonum: Dot) function
func TestDotOfVectors(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	testResult := 385.0
	v := mat.NewVecDense(len(data), data)
	u := mat.NewVecDense(len(data), data)
	resultTest := dotOfVectors(v, u)
	if resultTest != testResult {
		t.Fatal("Умножение выполнено с ошибкой.")
	}
}

// BenchmarkCreateRandomVector testing speed and memory use of
// createRandomVector function
func BenchmarkCreateRandomVector(b *testing.B) {
	for key, value := range mapVecTest {
		b.Run(key, benchCreateRandomVectorFunc(value))
	}
}

func benchCreateRandomVectorFunc(N int) func(b *testing.B) {
	return func(b *testing.B) {
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			v := createRandomVector(N)
			b.StopTimer()
			v.IsZero()
		}
	}
}

// BenchmarkScaleVector testing speed and memory use vector scaling
func BenchmarkScaleVector(b *testing.B) {
	for key, value := range mapVecTest {
		b.Run(key, benchScaleVectorFunc(value, 5.25))
	}
}

func benchScaleVectorFunc(N int, alpha float64) func(b *testing.B) {
	return func(b *testing.B) {
		v := createRandomVector(N)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			scaleVector(v, alpha)
			b.StopTimer()
			scaleVector(v, 1 / alpha)
		}
	}
}

// BenchmarkScaleVector testing speed and memory use vector scaling
func BenchmarkFrobeniusNormOfVector(b *testing.B) {
	for key, value := range mapVecTest {
		b.Run(key, benchFrobeniusNormOfVectorFunc(value))
	}
}

func benchFrobeniusNormOfVectorFunc(N int) func(b *testing.B) {
	return func(b *testing.B) {
		v := createRandomVector(N)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			f := frobeniusNormOfVector(v)
			b.StopTimer()
			_ = f
		}
	}
}

// BenchmarkAdditionOfVectors testing speed and memory use vector scaling
func BenchmarkAdditionOfVectors(b *testing.B) {
	for key, value := range mapVecTest {
		b.Run(key, benchAdditionOfVectorsFunc(value))
	}
}

func benchAdditionOfVectorsFunc(N int) func(b *testing.B) {
	return func(b *testing.B) {
		v := createRandomVector(N)
		u := createRandomVector(N)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			w := additionOfVectors(v, u, 0.0)
			b.StopTimer()
			_ = w
		}
	}
}

// BenchmarkAdditionOfVectors testing speed and memory use of vector scaling
func BenchmarkSubtractOfVectors(b *testing.B) {
	for key, value := range mapVecTest {
		b.Run(key, benchSubtractOfVectorsFunc(value))
	}
}

func benchSubtractOfVectorsFunc(N int) func(b *testing.B) {
	return func(b *testing.B) {
		v := createRandomVector(N)
		u := createRandomVector(N)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			w := subtractOfVectors(v, u)
			b.StopTimer()
			_ = w
		}
	}
}

// BenchmarkDotOfVectors testing speed and memory use of vectors dot
func BenchmarkDotOfVectors(b *testing.B) {
	for key, value := range mapVecTest {
		b.Run(key, benchDotOfVectorsFunc(value))
	}
}

func benchDotOfVectorsFunc(N int) func(b *testing.B) {
	return func(b *testing.B) {
		v := createRandomVector(N)
		u := createRandomVector(N)
		b.ReportAllocs()
		b.N = 10
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StartTimer()
			w := dotOfVectors(v, u)
			b.StopTimer()
			_ = w
		}
	}
}
