#!/bin/bash

outputdir=ppopp16-cori-data

mkdir -p "$outputdir"

nodes="1,4,8,12,16,20,24,32,64"
timelimits="02:00:00,01:00:00,00:45:00,00:45:00,00:45:00,00:45:00,00:45:00,00:30:00,00:30:00"

# blur_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# bg_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# ll_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 )
# int_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 )
# cp_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 )
# resize_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 )
# wavelet_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# transpose_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# sobel_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )

widths=( 2000 4000 10000 20000 30000 40000 50000 100000 )

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/blur/halide_blur_distributed.cpp ../apps/blur/distributed_blur $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/sobel/sobel_distributed.cpp ../apps/sobel/distributed_sobel $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/transpose/transpose_distributed.cpp ../apps/transpose/distributed_transpose $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/bilateral_grid/bilateral_grid_distributed.cpp ../apps/bilateral_grid/distributed_bilateral_grid $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/local_laplacian/local_laplacian_distributed.cpp ../apps/local_laplacian/distributed_local_laplacian $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/interpolate/interpolate_distributed.cpp ../apps/interpolate/distributed_interpolate $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/camera_pipe/camera_pipe_distributed.cpp ../apps/camera_pipe/distributed_camera_pipe $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/resize/resize_distributed.cpp ../apps/resize/distributed_resize $w $w &
    sleep 0.5
done

for w in "${widths[@]}"; do
    ./scalability-test.py -f -n "$nodes" -t "$timelimits" -o "$outputdir" -s ../apps/wavelet/wavelet_distributed.cpp ../apps/wavelet/distributed_wavelet $w $w &
    sleep 0.5
done
