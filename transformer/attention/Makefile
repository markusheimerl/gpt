CC = clang
CXX = clang++
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
CXXFLAGS = -O3 -march=native -Wall -Wextra -std=c++17
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -lcudart -lcublasLt -lcudnn -lstdc++

test.out: attention.o test.o cudnn_att.o
	$(CC) attention.o test.o cudnn_att.o $(CUDALIBS) $(LDFLAGS) -o $@

attention.o: attention.c attention.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c attention.c -o $@

test.o: test.c attention.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c test.c -o $@

cudnn_att.o: cudnn_att.cpp attention.h
	$(CXX) $(CXXFLAGS) -c cudnn_att.cpp -o $@

run: test.out
	@time ./test.out

clean:
	rm -f *.out *.o *.csv *.bin