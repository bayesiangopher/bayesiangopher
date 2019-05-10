CXX = g++
INCLUDE_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
INC = -I $(INCLUDE_PATH)
CFLAGS = -O2
LIBS = -larmadillo
SOURCE_MAT = mat_test.cpp
SOURCE_VEC = vec_test.cpp

all: 
	$(CXX) $(SOURCE_MAT) $(INC) -o out_mat $(CFLAGS) $(LIBS)
	$(CXX) $(SOURCE_VEC) $(INC) -o out_vec $(CFLAGS) $(LIBS)

clean:
	rm out_mat
	rm out_vec 