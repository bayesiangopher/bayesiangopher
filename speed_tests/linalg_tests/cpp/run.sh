#!/bin/bash

make clean
bash requirements.sh
make
bash testcases_time.sh