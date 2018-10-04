all: find_best_split.so
		

find_best_split.so: find_best_split.c
		gcc -std=c11 -ffast-math -O3 -shared -fPIC find_best_split.c -o find_best_split.so

clean:
		rm -rf *.o *.so
