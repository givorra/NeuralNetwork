/*
 * main.cpp
 *
 *  Created on: 26 de dic. de 2015
 *      Author: gacel
 */

#include "NeuralNetwork.h"

using namespace std;

int main()
{

	//Main de prueba para funcion logica XOR
	int nInputs 			= 4;
	int inputsDimension 	= 2;
	int nLayers 			= 3;
	int p[3] 				= {inputsDimension, 2, 1};
	int trainIterations 	= 10000;
	double errorFactible	= 0.15;
	double convergenceRatio	= 0.0;

	double** inputs 		= new double*[nInputs];	// Filas = nº de entradas
	double** y 				= new double*[nInputs];

	for(int i = 0; i < nInputs; i++)
	{
		inputs[i] 	= new double[inputsDimension];	// Columnas x fila = nº de neuronas en capa de salida
		y[i] 		= new double[1];
	}

	inputs[0][0] 	= 0;
	inputs[0][1] 	= 0;
	y[0][0]			= 1;

	inputs[1][0] 	= 0;
	inputs[1][1] 	= 1;
	y[1][0]			= 0;

	inputs[2][0] 	= 1;
	inputs[2][1] 	= 0;
	y[2][0]			= 0;

	inputs[3][0] 	= 1;
	inputs[3][1] 	= 1;
	y[3][0]			= 1;


	NeuralNetwork nn(nLayers, p, trainIterations, errorFactible, convergenceRatio);// = new NeuralNetwork(3, p);
	nn.train(inputs, y, nInputs);

	double* output = nn.classify(inputs[0]);
	cout << " x -> {0 0} = " << (output[0] > 0.5) << endl;
	output = nn.classify(inputs[1]);
	cout << " x -> {0 1} = " << (output[0] > 0.5) << endl;
	output = nn.classify(inputs[2]);
	cout << " x -> {1 0} = " << (output[0] > 0.5) << endl;
	output = nn.classify(inputs[3]);
	cout << " x -> {1 1} = " << (output[0] > 0.5) << endl;
	/*
	int nInputs 			= 25;
	int inputsDimension 	= 1;
	int nLayers 			= 3;
	int p[3] 				= {inputsDimension, 2, 1};
	int trainIterations 	= 10000;
	double errorFactible	= 0.0000001;
	double convergenceRatio	= 0.0;

	double** inputs 		= new double*[nInputs];	// Filas = nº de entradas
	double** y 				= new double*[nInputs];

	for(int i = 0; i < nInputs; i++)
	{
		inputs[i] 	= new double[inputsDimension];	// Columnas x fila = nº de neuronas en capa de salida
		y[i] 		= new double[1];
	}
	inputs[0][0] 	= -23.234;
	y[0][0] 		= 1;

	inputs[1][0] 	= -55.234;
	y[1][0] 		= 1;

	inputs[2][0] 	= -87.234;
	y[2][0] 		= 1;

	inputs[3][0] 	= -11.234;
	y[3][0] 		= 1;

	inputs[4][0] 	= -14.234;
	y[4][0] 		= 1;

	inputs[5][0] 	= -99.234;
	y[5][0] 		= 1;

	inputs[6][0] 	= -123.234;
	y[6][0] 		= 1;

	inputs[7][0] 	= -43.234;
	y[7][0] 		= 1;

	inputs[8][0] 	= -54.234;
	y[8][0] 		= 1;

	inputs[9][0] 	= -55.234;
	y[9][0] 		= 1;

	inputs[10][0] 	= -432.234;
	y[10][0] 		= 1;

	inputs[11][0] 	= -121.234;
	y[11][0] 		= 1;

	inputs[12][0] 	= -32.234;
	y[12][0] 		= 1;

	inputs[13][0] 	= 567.234;
	y[13][0] 		= 0;

	inputs[14][0] 	= 597.234;
	y[14][0] 		= 0;

	inputs[15][0] 	= 798.234;
	y[15][0] 		= 0;

	inputs[16][0] 	= 987.234;
	y[16][0] 		= 0;

	inputs[17][0] 	= 678.234;
	y[17][0] 		= 0;

	inputs[18][0] 	= 587.234;
	y[18][0] 		= 0;

	inputs[19][0] 	= 888.234;
	y[19][0] 		= 0;

	inputs[20][0] 	= 985.234;
	y[20][0] 		= 0;

	inputs[21][0] 	= 1111.234;
	y[21][0] 		= 0;

	inputs[22][0] 	= 1178.234;
	y[22][0] 		= 0;

	inputs[23][0] 	= 878.234;
	y[23][0] 		= 0;

	inputs[24][0] 	= 1900.234;
	y[24][0] 		= 0;
	NeuralNetwork nn(nLayers, p, trainIterations, errorFactible, convergenceRatio);// = new NeuralNetwork(3, p);
while(true){
	nn.train(inputs, y, nInputs);
}*/
}
/*
void NeuralNetwork::train(double** inputs, double** y, int nInputs)
{
	bool trained = false;
	initWeights();
	initHmatrix(nInputs);
	while(!trained)
	{
		double diferenceCost	= 0.0;
		double lastCost 		= 0.0;

		for(int j = 0; j < 5; j++)
		{
			initDeltaAndD();
			for(int i = 0; i < nInputs; i++)
			{
				setInputs(inputs[i]);
				forwardPropagation();
				saveOutput(i);
				setOutputError(y[i]);
				backPropagation();
				accumulateGradient();
			}
			calculateDerivate(nInputs);
			// Aqui --> Gradient check
			updateWeights();

			// Calculos para detectar como varia el coste cada 5 iteraciones
			double cost = costFunction(y, nInputs);
			diferenceCost = cost - lastCost;
			lastCost = cost;

			cout << "Coste: " << cost << endl;
		}
		// Ver si se finaliza el training en funcion de como varia el coste cada 5 iteraciones
		trained = diferenceCost > 0.0001;
	}
}
*/
