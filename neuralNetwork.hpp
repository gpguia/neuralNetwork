#include <stdlib.h>
#include <vector>
#include <list>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <time.h>
#include <numeric>
#include <random>

using namespace std;

#define VD vector < vector <double > >
#define MP make_pair

int INPUT,HIDDEN,REPETITIONS;
double LEARN_RATE;
int OUTPUT = 1;
int LAYERS = 2;

int SEED = 1;
int FSEED = 1;

int LIMIT = 100000;

int BATCH = 200;
int TODO = 1000;

int FACTOR = 2, FACTOR2 = 2;


int RANDPERIOD = 1;


bool WITHBIAS = true;
//just for debug stuff
bool PRINT = true;

long sum, conv;
double maxerr;
int numExp,sizeBatch;


void readInput();
void readParam();
void init(int n);
void clear();
double activeFunc(double x);
double dActiveFunc(double x);
void propagation(int x);
void backPropagation(int x);
void upWeight();
void printNet();
void train(int epo, int k);
void genInputs();

vector<int> nodeLayer;
vector<double> out;
vector< VD > weight;
VD in,err,value,bias,as;