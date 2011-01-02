#!/bin/sh

for i in {1..10}; do 
build/evolver --gpu --seed 0 | tee archive/evol_gpu_0_$i.log;
build/evolver --cpu --seed 0 | tee archive/evol_cpu_0_$i.log;
done

for i in {1..10}; do 
build/evolver --gpu --seed 1 | tee archive/evol_gpu_1_$i.log;
build/evolver --cpu --seed 1 | tee archive/evol_cpu_1_$i.log;
done

for i in {1..10}; do 
build/evolver --gpu --seed 2 | tee archive/evol_gpu_2_$i.log;
build/evolver --cpu --seed 2 | tee archive/evol_cpu_2_$i.log;
done

for i in {1..10}; do 
build/evolver --gpu --seed 3 | tee archive/evol_gpu_3_$i.log;
build/evolver --cpu --seed 3 | tee archive/evol_cpu_3_$i.log;
done