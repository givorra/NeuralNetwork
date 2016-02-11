#ifndef _NeuralNetwork_H_
#define _NeuralNetwork_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <fstream>

using namespace std;

struct Neuron {
  double  _a;     // salida
  double  _e;     // error 
  double* _w;     // pesos  
};

struct Layer {
  int     	_nNeurons;  // Numero neuronas de la capa
  Neuron*	_neurons;   // Neuronas
  double** 	_D;
  double**	_Delta;
};

class NeuralNetwork
{
private:
	Layer*		_layers;
	int			_nLayers;
	double**	_h;					// Valores de la capa de salida para cada ejemplo del training

	double		_errorFactible;		// Error aceptado
	int 		_trainIterations;	// Iteraciones maximas
	double		_convergenceRatio;	// Ratio de convergencia

	void initWeights();									// Inicializa los pesos con random
	void initDeltaAndD();								// Inicializa matrices D y Delta de cada capa
	void forwardPropagation();							// Forward propagation para toda la red
	void setInputs(double* inputs);						// Introduce las entradas en la capa de entrada
	void h(int layer, int neuron);						// Devuelve la activación de una neurona
	void setOutputError(double* y);						// Calcula error de la capa de salida
	void backPropagation();								// Calcula el error de cada neurona de cada capa
	void accumulateGradient();							// Calcula Delta(l)_ij += a(l)_j*e(l+1)_i
	void calculateDerivate(int m);
	void updateWeights();
	double costFunction(double** y, int nInputs);
	void initHmatrix(int nInputs);						// Inicializa la matriz de salidas con tamaño adecuado
	void saveOutput(int i);								// Guarda la salida de un ejemplo del training
	double* getOutput();
	void normalizeInputs(double** inputs, int nInputs);

public:
	NeuralNetwork(int nLayers, int nNeuronXLayer[], int trainIterations, double errorFactible, double convergenceRetio); // Por ejemplo NeuralNetwork(3, {2,3,1})
	//~NeuralNetwork();
	double* classify(double* x);							// Clasifica una entrada
	void train(double** inputs, double** y, int nInputs);	// Entrena la red con las entradas recibidas
};

#endif

