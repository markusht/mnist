CC=g++
CFLAGS=-I.
DEPS = csv_util.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm *.o

main: main.o
	$(CC) -o ../out/main main.o
