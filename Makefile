grid: grid.cpp
	g++ grid.cpp -o grid `pkg-config --cflags --libs opencv` -lpthread