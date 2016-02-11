#include "NeuralNetwork.h"

const bool debug = false;

using namespace std;

void initRandoms()
{
  srand((unsigned)time(NULL));
}

double randomDouble(double min, double max)
{
  return ((double) rand() / RAND_MAX) * (max-min) + min;
}


NeuralNetwork::NeuralNetwork(int nLayers, int nNeuronXLayer[], int trainIterations, double errorFactible, double convergenceRetio)
{
	initRandoms();

	_trainIterations 	= trainIterations;
	_errorFactible 		= errorFactible;
	_convergenceRatio	= convergenceRetio;

	if(nLayers >= 2)
	{
		_nLayers = nLayers;
		_layers  = new Layer[_nLayers];

		for(int i = 0; i < _nLayers; i++)	// Inicializamos cada capa
		{
			if(i != _nLayers-1)	// Si no es la capa de salida añadimos neurona para valor de oscinacion
			{
				_layers[i]._nNeurons = nNeuronXLayer[i]+1;
			}
			else 
			{
				_layers[i]._nNeurons = nNeuronXLayer[i];
			}
			// Matriz Delta y D, si no es la capa de entrada
			if(i > 0)
			{
				// Filas
				_layers[i]._D 		= new double*[_layers[i]._nNeurons];
				_layers[i]._Delta 	= new double*[_layers[i]._nNeurons];

				// Columnas por fila
				for(int j = 0; j < _layers[i]._nNeurons; j++)
				{
					_layers[i]._D[j] 		= new double[_layers[i-1]._nNeurons];
					_layers[i]._Delta[j] 	= new double[_layers[i-1]._nNeurons];
				}
			}

			_layers[i]._neurons = new Neuron[_layers[i]._nNeurons];	

			for(int j = 0; j < _layers[i]._nNeurons; j++)	// Inicializamos las neuronas de cada capa
			{
				_layers[i]._neurons[j]._a = 1;
				_layers[i]._neurons[j]._e = 0;

				if(i > 0)	// Si no es la primera capa, creamos vector de pesos con el tamaño del numero de neuronas de la capa anterior
				{
					_layers[i]._neurons[j]._w 		= new double[_layers[i-1]._nNeurons];
				}
				else		// Si no, a nulo ya que la capa de entrada no tiene pesos
				{
					_layers[i]._neurons[j]._w 		= NULL;
				}
			}
		}
	}
	else
	{
		cerr << "Error: como mínimo debe haber una capa de entrada y una de salida." << endl;
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		cout << "### OPEN Debug constructor ###" << endl;
		for(int i = 0; i < _nLayers; i++)
		{
			cout << "	> Capa: " << i << endl;
			cout << "  		-Nº de neuronas: " << _layers[i]._nNeurons << endl;
			cout << "		-Neuronas:" << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				cout << "			> Neurona nº " << j << endl;
				cout << "				- a = " << _layers[i]._neurons[j]._a << endl;
				cout << "				- e = " << _layers[i]._neurons[j]._e << endl;
				if(i > 0)
				{
					for(int k = 0; k < _layers[i-1]._nNeurons; k++)
					{
						cout << "				- w["<<k<<"]= " << _layers[i]._neurons[j]._w[k] << endl;
					}
				}
			}
		}
		cout << "### CLOSE Debug constructor ###" << endl;
	}
}
/*
NeuralNetwork::~NeuralNetwork()
{

	for(int i = 0; i < _nLayers; i++)	// Inicializamos cada capa
	{

		for(int j = 0; j < _layers[i]._nNeurons; j++)	// Inicializamos las neuronas de cada capa
		{
			_layers[i]._neurons[j]._a = 1;
			_layers[i]._neurons[j]._e = 0;

			if(i > 0)	// Si no es la primera capa, creamos vector de pesos con el tamaño del numero de neuronas de la capa anterior
			{
				delete[] _layers[i]._neurons[j]._w;
			}
		}

		delete[] _layers[i]._neurons;

		if(i > 0)
		{
			// Columnas por fila
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				delete[] _layers[i]._D[j];
				delete[] _layers[i]._Delta[j];
			}

			// Filas
			delete[] _layers[i]._D;
			delete[] _layers[i]._Delta;
		}

		delete[] _layers;
	}
}
*/
void NeuralNetwork::initWeights()
{
	for(int i = 1; i < _nLayers; i++)	// Para cada capa
	{
		for(int j = 0; j < _layers[i]._nNeurons; j++)	// Para cada neurona de cada capa
		{
			if(j == 0 && i != _nLayers-1)	// Si es neurona correspondiente a unidad de oscilacion, la saltamos
				continue;
			//{
				for(int k = 0; k < _layers[i-1]._nNeurons; k++)		// Inicializa pesos de la neurona
				{
					_layers[i]._neurons[j]._w[k] = randomDouble(-0.5, 0.5);
				}
			//}
		}
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		cout << "### OPEN Debug initWeights ###" << endl;
		for(int i = 0; i < _nLayers; i++)
		{
			cout << "	> Capa: " << i << endl;
			cout << "  		-Nº de neuronas: " << _layers[i]._nNeurons << endl;
			cout << "		-Neuronas:" << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				cout << "			> Neurona nº " << j << endl;
				cout << "				- a = " << _layers[i]._neurons[j]._a << endl;
				cout << "				- e = " << _layers[i]._neurons[j]._e << endl;
				if(i > 0)
				{
					for(int k = 0; k < _layers[i-1]._nNeurons; k++)
					{
						cout << "				- w["<<k<<"]= " << _layers[i]._neurons[j]._w[k] << endl;
					}
				}
			}
		}
		cout << "### CLOSE Debug initWeights ###" << endl;
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLOSE DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
}

void NeuralNetwork::forwardPropagation()
{
	for(int layer = 1; layer < _nLayers; layer++)	// Para cada capa
	{
		for(int neuron = 0; neuron < _layers[layer]._nNeurons; neuron++)	// Para cada neurona de cada capa
		{
			if(neuron == 0 && layer != _nLayers-1)	// Si es neurona correspondiente al umbral, la saltamos
				continue;//MODIFICADO
			h(layer,neuron);	// Calcular su activación en base a las salidas de la capa anterior
		}
	}

	if(debug)
	{
		cout << "### OPEN Debug forwardPropagation ###" << endl;
		for(int i = 0; i < _nLayers; i++)
		{
			cout << "	> Capa: " << i << endl;
			cout << "  		-Nº de neuronas: " << _layers[i]._nNeurons << endl;
			cout << "		-Neuronas:" << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				cout << "			> Neurona nº " << j << endl;
				cout << "				- a = " << _layers[i]._neurons[j]._a << endl;
				cout << "				- e = " << _layers[i]._neurons[j]._e << endl;
				if(i > 0)
				{
					for(int k = 0; k < _layers[i-1]._nNeurons; k++)
					{
						cout << "				- w["<<k<<"]= " << _layers[i]._neurons[j]._w[k] << endl;
					}
				}
			}
		}
		cout << "### CLOSE Debug forwardPropagation ###" << endl;
	}
}

void NeuralNetwork::backPropagation()
{
	if(debug)
	{
		cout << "### OPEN Debug backPropagation ###" << endl;
		cout << "Calculos para propagacion del error hacia atrás"<<endl;
	}
	for(int i = _nLayers-2; i > 0; i--)
	{
		if(debug)
			cout << "Capa "<<i<<": "<<endl;

		for(int j = 1; j < _layers[i]._nNeurons; j++)
		{
			_layers[i]._neurons[j]._e = 0;

			for(int k = 0; k < _layers[i+1]._nNeurons; k++)	// El error de la unidad de oscinacion siempre debe ser cero, por lo que puedo contarlo (necesario para penultima capa)
			{
				_layers[i]._neurons[j]._e += (_layers[i+1]._neurons[k]._w[j] * _layers[i+1]._neurons[k]._e);

				if(debug)
					cout << "Error_"<<j<<" += "<< _layers[i+1]._neurons[k]._w[j] <<" * "<< _layers[i+1]._neurons[k]._e<<endl;
			}
			_layers[i]._neurons[j]._e = _layers[i]._neurons[j]._e * (_layers[i]._neurons[j]._a * (1 - _layers[i]._neurons[j]._a));
		}
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		//cout << "### OPEN Debug backPropagation ###" << endl;
		cout <<endl<<endl;
		for(int i = 0; i < _nLayers; i++)
		{
			cout << "	> Capa: " << i << endl;
			cout << "  		-Nº de neuronas: " << _layers[i]._nNeurons << endl;
			cout << "		-Neuronas:" << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				cout << "			> Neurona nº " << j << endl;
				cout << "				- a = " << _layers[i]._neurons[j]._a << endl;
				cout << "				- e = " << _layers[i]._neurons[j]._e << endl;
				if(i > 0)
				{
					for(int k = 0; k < _layers[i-1]._nNeurons; k++)
					{
						cout << "				- w["<<k<<"]= " << _layers[i]._neurons[j]._w[k] << endl;
					}
				}
			}
		}
		cout << "### CLOSE Debug backPropagation ###" << endl;
	}
}

void NeuralNetwork::accumulateGradient()
{

	// DEBUG
	if(debug)
	{
		cout << "### OPEN Debug accumulateGradient ###" << endl;
		cout << "Calculos para acumulación del gradiente" << endl;
	}
	for(int i = 1; i < _nLayers; i++)	// Para cada capa
	{
		if(debug)
			cout << "Capa "<<i<<":"<<endl;

		for(int j = 0; j < _layers[i]._nNeurons; j++)	// Para cada neurona de la capa
		{
			for(int k = 0; k < _layers[i-1]._nNeurons; k++)	// Para cada peso de la neurona (nº de entradas/neuronas de la capa anterior)
			{
				_layers[i]._Delta[j][k] += _layers[i-1]._neurons[k]._a * _layers[i]._neurons[j]._e;

				if(debug)
					cout << "Delta_"<<j<<k<<" += "<< _layers[i-1]._neurons[k]._a << " * " <<_layers[i]._neurons[j]._e << " = " << _layers[i]._Delta[j][k] << endl;
			}
		}
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		//cout << "### OPEN Debug accumulateGradient ###" << endl;
		cout <<endl<<endl;
		for(int i = 1; i < _nLayers; i++)	// Para cada capa
		{
			cout << "	> Capa nº " << i << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)	// Para cada neurona de la capa
			{
				cout << "		";
				for(int k = 0; k < _layers[i-1]._nNeurons; k++)	// Para cada peso de la neurona (nº de entradas/neuronas de la capa anterior)
				{
					cout << _layers[i]._Delta[j][k] << "	";
				}
				cout << endl;
			}
		}
		cout << "### CLOSE Debug accumulateGradient ###" << endl;
	}

}

void NeuralNetwork::calculateDerivate(int m)
{
	// DEBUG
	if(debug)
	{
		cout << "### OPEN Debug calculateDerivate ###" << endl;
		cout << "Calculos de la derivada: " << endl;
	}
	for(int i = 1; i < _nLayers; i++)	// Para cada capa
	{
		if(debug)
			cout << "Capa "<<i<<": " << endl;

		for(int j = 0; j < _layers[i]._nNeurons; j++)	// Para cada neurona de la capa
		{
			for(int k = 0; k < _layers[i-1]._nNeurons; k++)	// Para cada peso de la neurona (nº de entradas/neuronas de la capa anterior)
			{
				_layers[i]._D[j][k] = (1.0/m)*_layers[i]._Delta[j][k];

				if (debug)
					cout << "D_"<<j<<k<<" = " << "1/"<<m<<" * " << _layers[i]._Delta[j][k] << endl;
			}
		}
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		//cout << "### OPEN Debug calculateDerivate ###" << endl;
		cout <<endl<<endl;
		for(int i = 1; i < _nLayers; i++)	// Para cada capa
		{
			cout << "	> Capa nº " << i << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)	// Para cada neurona de la capa
			{
				cout << "		";
				for(int k = 0; k < _layers[i-1]._nNeurons; k++)	// Para cada peso de la neurona (nº de entradas/neuronas de la capa anterior)
				{
					cout << _layers[i]._D[j][k] << "	";
				}
				cout << endl;
			}
		}
		cout << "### CLOSE Debug calculateDerivate ###" << endl;
	}
}

void NeuralNetwork::updateWeights()
{
	if(debug)
	{
		cout << "### OPEN Debug updateWeights ###" << endl;
		cout << "Calculos actualizacion de pesos:" << endl;
	}
	for(int i = 1; i < _nLayers; i++)	// Para cada capa
	{
		if(debug)
			cout << "Capa "<<i<<": " << endl;

		for(int j = 0; j < _layers[i]._nNeurons; j++)	// Para cada neurona de la capa
		{
			//***************
			if(i != _nLayers-1 && j==0)
				continue;
			//***************

			for(int k = 0; k < _layers[i-1]._nNeurons; k++)	// Para cada peso de la neurona (nº de entradas/neuronas de la capa anterior)
			{
				_layers[i]._neurons[j]._w[k] -= _layers[i]._D[j][k]; // *ALPHA ;

				if(debug)
					cout << "W_" << j << k << " += " << _layers[i]._D[j][k] << " = " << _layers[i]._neurons[j]._w[k] << endl;
			}
		}
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		//cout << "### OPEN Debug updateWeights ###" << endl;
		cout <<endl<<endl;
		for(int i = 0; i < _nLayers; i++)
		{
			cout << "	> Capa: " << i << endl;
			cout << "  		-Nº de neuronas: " << _layers[i]._nNeurons << endl;
			cout << "		-Neuronas:" << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				cout << "			> Neurona nº " << j << endl;
				cout << "				- a = " << _layers[i]._neurons[j]._a << endl;
				cout << "				- e = " << _layers[i]._neurons[j]._e << endl;
				if(i > 0)
				{
					for(int k = 0; k < _layers[i-1]._nNeurons; k++)
					{
						cout << "				- w["<<k<<"]= " << _layers[i]._neurons[j]._w[k] << endl;
					}
				}
			}
		}
		cout << "### CLOSE Debug updateWeights ###" << endl;
	}
}

void NeuralNetwork::h(int layer, int neuron)
{
	double sum = 0;

	for(int i = 0; i < _layers[layer-1]._nNeurons; i++)	// Funcion h_thetha(x)
		sum += _layers[layer]._neurons[neuron]._w[i] * _layers[layer-1]._neurons[i]._a;

	_layers[layer]._neurons[neuron]._a = 1.0 / (1.0 + exp(-1.0 * sum));	// Activación sigmoidea
}

void NeuralNetwork::setInputs(double* input)
{
	for(int i = 1; i < _layers[0]._nNeurons; i++)
	{
		_layers[0]._neurons[i]._a = input[i-1];
	}
}

void NeuralNetwork::initDeltaAndD()
{
	for(int i = 1; i < _nLayers; i++)
	{
		for(int j = 0; j < _layers[i]._nNeurons; j++)
		{
			for(int k = 0; k < _layers[i-1]._nNeurons; k++)
			{
				_layers[i]._D[j][k] 	= 0;
				_layers[i]._Delta[j][k] = 0;
			}
		}
	}
}

void NeuralNetwork::setOutputError(double* y)
{
	if(debug)
	{
			cout << "### OPEN Debug setOutputError ###" << endl;
			cout << "Calculos de error de la salida: " << endl;
	}
	for(int i = 0; i < _layers[_nLayers-1]._nNeurons; i++)
	{
		_layers[_nLayers-1]._neurons[i]._e = _layers[_nLayers-1]._neurons[i]._a - y[i];	// error = a(l)_i - y_i

		if(debug)
			cout << "e_"<<i<<" ="<< _layers[_nLayers-1]._neurons[i]._a  << " - " << y[i] << " = " << _layers[_nLayers-1]._neurons[i]._e;
	}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OPEN DEBUG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	if(debug)
	{
		//cout << "### OPEN Debug setOutputError ###" << endl;
		cout <<endl<<endl;
		for(int i = 0; i < _nLayers; i++)
		{
			cout << "	> Capa: " << i << endl;
			cout << "  		-Nº de neuronas: " << _layers[i]._nNeurons << endl;
			cout << "		-Neuronas:" << endl;
			for(int j = 0; j < _layers[i]._nNeurons; j++)
			{
				cout << "			> Neurona nº " << j << endl;
				cout << "				- a = " << _layers[i]._neurons[j]._a << endl;
				cout << "				- e = " << _layers[i]._neurons[j]._e << endl;
				if(i > 0)
				{
					for(int k = 0; k < _layers[i-1]._nNeurons; k++)
					{
						cout << "				- w["<<k<<"]= " << _layers[i]._neurons[j]._w[k] << endl;
					}
				}
			}
		}
		cout << "### CLOSE Debug setOutputError ###" << endl;
	}
}

void NeuralNetwork::saveOutput(int i)
{
	for(int j = 0; j < _layers[_nLayers-1]._nNeurons; j++)
		_h[i][j] = _layers[_nLayers-1]._neurons[j]._a;
}

double NeuralNetwork::costFunction(double** y, int nInputs)
{
	double cost = 0;
	for(int i = 0; i < nInputs; i++)
	{
		for(int j = 0; j < _layers[_nLayers-1]._nNeurons; j++)
		{
			cost += y[i][j] * log(_h[i][j]) + (1-y[i][j]) * log(1-_h[i][j]);
		}
	}
	return cost*(-1.0/nInputs);
}

void NeuralNetwork::initHmatrix(int nInputs)
{
	_h = new double*[nInputs];	// Filas = nº de entradas

	for(int i = 0; i < nInputs; i++)
		_h[i] = new double[_layers[_nLayers-1]._nNeurons];	// Columnas x fila = nº de neuronas en capa de salida
}

void NeuralNetwork::train(double** inputs, double** y, int nInputs)
{
	bool trained = false;
	initWeights();
	initHmatrix(nInputs);
	//normalizeInputs(inputs, nInputs);

	double diferenceCost;
	double lastCost	= 1.0;

	int i=0;	// Guardo la i para saber su valor al salir del for

	while((i < _trainIterations) && (!trained) && (lastCost >= _errorFactible))	// Maximo _trainIterations iteraciones, el train ya no mejora, el error actual es factible
	{
		diferenceCost	= 0.0;
		lastCost 		= 0.0;

		for(int j = 0; j < 5; j++ && i++)
		{
			initDeltaAndD();
			for(int k = 0; k < nInputs; k++)
			{
				setInputs(inputs[k]);
				forwardPropagation();
				saveOutput(k);
				setOutputError(y[k]);
				backPropagation();
				accumulateGradient();
			}
			calculateDerivate(nInputs);
			// Aqui --> Gradient check
			updateWeights();

			// Calculos para detectar como varia el coste cada 5 iteraciones
			double cost = costFunction(y, nInputs);
			if(j>0)
				diferenceCost += lastCost-cost;

			lastCost = cost;

			//cout << "Coste: " << cost << endl;
		}
		// Ver si se finaliza el training en funcion de como varia el coste cada 5 iteraciones
		trained = diferenceCost < _convergenceRatio;
	}

	// Resultados del entrenamiento
	cout << "# Resumen del training: " << endl;
	cout << " -Iteraciones realizadas: "<< i<<endl;
	cout << " -Error de clasificación: "<<lastCost<<endl;
	if(trained)
		cout << " -*El training se ha detenido debido a que el error ya no se reduce." << endl;
}

double* NeuralNetwork::getOutput()
{
	double* output = new double[_layers[_nLayers-1]._nNeurons];

	for(int i = 0; i < _layers[_nLayers-1]._nNeurons; i++)
	{
		output[i] = _layers[_nLayers-1]._neurons[i]._a;
	}

	return output;
}

double* NeuralNetwork::classify(double* x)
{
	setInputs(x);
	forwardPropagation();
	return getOutput();
}

void NeuralNetwork::normalizeInputs(double** inputs, int nInputs)
{
	double total[_layers[0]._nNeurons-1];

	for(int i = 0; i < _layers[0]._nNeurons-1; i++)
	{
		total[i] = 0;
		for(int j = 0; j < nInputs; j++)
		{
			total[i] += inputs[j][i];
		}
	}
	for(int i = 0; i < _layers[0]._nNeurons-1; i++)
	{
		for(int j = 0; j < nInputs; j++)
		{
			inputs[j][i] = inputs[j][i]/total[i];
		}
	}
}
