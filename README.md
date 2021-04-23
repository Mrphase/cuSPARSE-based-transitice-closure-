# cuSPARSE-based-transitice-closure

## prepare data
![image](https://github.com/Mrphase/cuSPARSE-based-transitice-closure-/blob/master/img/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202021-04-23%20010124.JPG)
```
./text_to_bin.bin ./toy_graph/mc2depi.mtx 1 0
```
the first number '1' means we want to reverse the edge, the second number '0' means the number of lins we want to skip

## compile
```
nvcc -o kernel.out kernel2.cu -std=c++11 -lcusparse -O3
```

## run
```
./kernel.out delaunay_n14.txt_beg_pos.bin delaunay_n14.txt_csr.bin delaunay_n14.txt_weight.bin 
```
