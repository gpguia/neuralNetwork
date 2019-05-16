OBJS=neuralNetwork.o
CC=g++
PROGRAM=./neuralNet
CFLAGS=-std=c++11
INDIR=./inputs/

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CC) -o $(PROGRAM) $(OBJS)
clean:
	rm -f $(PROGRAM) $(OBJS)

1: $(PROGRAM)
	$(PROGRAM) $(INDIR)in3.txt 3
	
2: $(PROGRAM)
	$(PROGRAM) $(INDIR)in4.txt 4
	
3: $(PROGRAM)
	$(PROGRAM) $(INDIR)in5.txt 5