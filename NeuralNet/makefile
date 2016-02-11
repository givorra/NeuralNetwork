#Pon aquí tus archivos cpp, cc, ect...
SOURCE_FILES = NeuralNetwork.h NeuralNetwork.cpp main.cpp
#Pon aquí tus archivos cpp, cc, ect... pero con la extensión .o
SOURCE_OBJECTS = NeuralNetwork.o main.o
#Aquí el compilador que vas a usar
COMP = g++
#Aquí el nombre y directorio al que irá el ejecutable
OUT_FILE = ./main

all: 
	$(COMP) -c $(SOURCE_FILES)
	$(COMP) -o $(OUT_FILE) $(SOURCE_OBJECTS)