#!/bin/bash

outputdir=ppopp16-data

mkdir -p "$outputdir"

nodes="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"

# blur_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# bg_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# ll_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 )
# int_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 )
# cp_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 )
# resize_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 )
# wavelet_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# transpose_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )
# sobel_widths=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 )

blur_widths=( 2000 4000 6000 8000 10000 14000 15000 18000 20000 22000 23000 )
bg_widths=( 2000 4000 6000 8000 10000 14000 15000 18000 20000 22000 23000 )
ll_widths=( 2000 4000 6000 8000 10000 14000 15000 )
int_widths=( 2000 4000 6000 8000 10000 )
cp_widths=( 2000 4000 6000 8000 10000 14000 15000 23000 )
resize_widths=( 2000 4000 6000 8000 10000 )
wavelet_widths=( 2000 4000 6000 8000 10000 14000 15000 18000 20000 22000 23000 )
transpose_widths=( 2000 4000 6000 8000 10000 14000 15000 18000 20000 22000 23000 )
sobel_widths=( 2000 4000 6000 8000 10000 14000 15000 18000 20000 22000 23000 )

for w in "${blur_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/blur/halide_blur_distributed.cpp ../apps/blur/distributed_blur $w $w
done

for w in "${bg_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/bilateral_grid/bilateral_grid_distributed.cpp ../apps/bilateral_grid/distributed_bilateral_grid $w $w
done

for w in "${ll_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/local_laplacian/local_laplacian_distributed.cpp ../apps/local_laplacian/distributed_local_laplacian $w $w
done

for w in "${int_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/interpolate/interpolate_distributed.cpp ../apps/interpolate/distributed_interpolate $w $w
done

for w in "${cp_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/camera_pipe/camera_pipe_distributed.cpp ../apps/camera_pipe/distributed_camera_pipe $w $w
done

for w in "${resize_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/resize/resize_distributed.cpp ../apps/resize/distributed_resize $w $w
done

for w in "${wavelet_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/wavelet/wavelet_distributed.cpp ../apps/wavelet/distributed_wavelet $w $w
done

for w in "${transpose_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/transpose/transpose_distributed.cpp ../apps/transpose/distributed_transpose $w $w
done

for w in "${sobel_widths[@]}"; do
    ./scalability-test.py -n "$nodes" -o "$outputdir" -s ../apps/sobel/sobel_distributed.cpp ../apps/sobel/distributed_sobel $w $w
done
