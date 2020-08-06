# cuSPARSE-based-transitice-closure

##prepare data
```
./text_to_bin.bin ./toy_graph/mc2depi.mtx 1 0
```
the first number '1' means we want to reverse the edge, the second number '0' means the number of lins we want to skip

##compile
```
nvcc -o kernel.out kernel2.cu -std=c++11 -lcusparse -O3
```

##run
```
./kernel.out delaunay_n14.txt_beg_pos.bin delaunay_n14.txt_csr.bin delaunay_n14.txt_weight.bin 
```
