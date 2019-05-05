#!/bin/bash

repeats=10
# vector tests
echo -e "Vector tests"
for ((i=1; i<=6; i++))
do
    for j in 1024 16384 65536 131072 262144 524288
    do
        ./out_vec $i $repeats $j
    done
done

# matrix tests
echo -e "\nMatrix tests"
for ((i=1; i<=10; i++))
do
    for j in 32 128 256 512 1024
    do
        ./out_mat $i $repeats $j
    done
done

