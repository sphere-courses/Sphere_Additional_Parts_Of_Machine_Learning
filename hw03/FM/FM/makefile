all: FM.so
		

FM.so: FM.c
		gcc -std=c11 -ffast-math -O3 -shared -fPIC FM.c -o FM.so

clean:
		rm -rf *.o *.so
