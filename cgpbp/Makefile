# GCC complier 
CC=gcc

# GCC complier options
# -pedantic		Issue all the warnings demanded by strict ISO C
# -Wall			Turns on all optional warnings which are desirable for normal code.
# -O3         	Turns on all optimizations
# -g			Turns on debugging information	
# -pg			Turns on profiling information (for gprof)
# -w			Ignore warning 
# -fpermissive
# -std=c++11    Use c++ 2011 standard
	
CFLAGS= -I ../src/include -L ../src/ 
LDFLAGS= -pedantic -Wall -O3 -fopenmp -lfann -lm -std=c11

main: main.c cgpbplib.c cgpbplib.h 
	@$(CC) $(CFLAGS) -o main main.c cgpbplib.c $(LDFLAGS) 
