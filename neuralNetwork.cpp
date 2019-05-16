#include "neuralNetwork.hpp"


int main(int argc, char *argv[]){
		
	cout << endl;
	cout << "Please type the number of input nodes: ";
	cin >> INPUT;
	cout << "Please type the number of hidden nodes: ";
	cin >> HIDDEN;
	cout << "Please type the number of repetitions: ";
	cin >> REPETITIONS;
	cout << "Please type the Learning Rate: ";
	cin >> LEARN_RATE;
	
	genInputs();
	
	numExp = (int)pow(2,INPUT);
	sizeBatch = (int)(0.75*numExp);
	nodeLayer.push_back(INPUT);
	nodeLayer.push_back(HIDDEN);
	nodeLayer.push_back(OUTPUT);
	
	sum = 0;
	conv = 0;
	long start,stop;
	double timer = 0;
	int prev = 0;
	for(int i = SEED; i < REPETITIONS + SEED; i++){
		init((i+1)*(FSEED+1));
		start = clock();
		train(LIMIT,i-SEED);
		stop = clock();
		if(conv != prev)
			timer += (stop - start) / double(CLOCKS_PER_SEC)*1000;
		prev = conv;
		clear();
	}
	
	cout << "Number of epochs: " << sum*1.0/conv << endl; 
	cout << "Success: " << conv << endl;
	cout << "Timer: " << timer << endl;
	
	return 0;	
}

void readInput(){
	//TODO
}

void genInputs(){
	list< pair<string,int> > lst;
	list< pair<string,int> >::iterator it;
	lst.push_back(MP("",0));
	
	int i;
	string s;
	int size = (int)lst.front().first.size();
	while(size != INPUT){
		s = lst.front().first;
		i = lst.front().second;
		lst.pop_front();
		lst.push_back(MP(s+"0",i));
		lst.push_back(MP(s+"1",i+1));
		size = (int)lst.front().first.size();
	}
	it = lst.begin();
	for(;it!=lst.end();it++){
		in.push_back(vector<double>());
		for(int j=0;j<INPUT;j++)
			in.back().push_back(double((*it).first[j]-'0'));
		if((*it).second%2)
			out.push_back(1);
		else
			out.push_back(0);
	}
}

void init(int n){
	srand(n);
	double d;
	weight.push_back(VD());
	for(int l = 1; l < LAYERS + 1;l++){
		weight.push_back(VD());
		for(int i=0;i<nodeLayer[l];i++){
			weight[l].push_back(vector<double>());
			for(int j=0;j<nodeLayer[l-1];j++){
				d = (double)rand() / RAND_MAX;
				weight[l][i].push_back(-1 + d*2);
			}
		}
	}
	
	for(int l = 0; l < LAYERS + 1; l++){
		value.push_back(vector<double>());
		as.push_back(vector<double>());
		err.push_back(vector<double>());
		bias.push_back(vector<double>());
		for(int i=0; i < nodeLayer[l];i++){
			value[l].push_back(0);
			as[l].push_back(0);
			err[l].push_back(0);
			d = (double)rand() / RAND_MAX;
			if(WITHBIAS)
				bias[l].push_back(-1 + d*2);
		}
	}
}

void clear(){
	weight.clear();
	value.clear();
	as.clear();
	err.clear();
	bias.clear();
}

double activeFunc(double x){
	return 1/(1 + exp(-x));
}

double dActiveFunc(double x){
	return x*(1-x);	
}

void propagation(int x){
	for(int i=0; i < INPUT; i++){
		value[0][i] = in[x][i];
		as[0][i] = value[0][i];	
	}
	for(int l = 1; l < LAYERS + 1; l++){
		for(int i=0; i < nodeLayer[l]; i++){
			if(WITHBIAS)
				value[l][i] = bias[l][i];
			else
				value[l][i] = 0;
			for(int j = 0; j< nodeLayer[l-1];j++)
				value[l][i] += as[l-1][j] * weight[l][i][j];
			as[l][i] = activeFunc(value[l][i]);
		}
	}
}

void backPropagation(int x){
	for(int i=0; i < OUTPUT; i++){
		err[LAYERS][i] = (-as[LAYERS][i]+out[x]) * 	dActiveFunc(as[LAYERS][i]);
		if(abs(out[x]-as[LAYERS][0]) > maxerr)
			maxerr = abs(out[x]-as[LAYERS][0]);	
	}	
	double erro;
	for(int l = LAYERS - 1; l >= 0 ;l--){
		for(int i=0;i<nodeLayer[l];i++){
			erro = 0;
			for(int j=0;j<nodeLayer[l+1];j++){
			erro += err[l+1][j] * weight[l+1][j][i];
			}
			err[l][i] = erro * dActiveFunc(as[l][i]);
		}	
	}
}

void upWeight(){
	for(int l=1;l<LAYERS+1;l++){
		for(int i=0;i<nodeLayer[l];i++){
			for(int j=0;j<nodeLayer[l-1];j++){
				weight[l][i][j] += LEARN_RATE * as[l-1][j] * err[l][i];
			}	
			if(WITHBIAS)
				bias[l][i] += LEARN_RATE * err[l][i];
		}	
	}
}

void printNet(){
	for(int x = 0;x < numExp;x++){
		propagation(x);
	}
	for(int l=1; l < LAYERS + 1; l++){
		cout << "Layer: " << l << endl;	
		for(int n = 0; n < nodeLayer[l];n++){
			cout << " node: " << n << endl;
			for(int k = 0; k < nodeLayer[l-1];k++){
				cout << " " << k <<" weigth: " << weight[l][n][k] << endl;
			}
		}
	}
	cout << endl;
}

void train(int epo, int k){
	vector<int> p;
	for(int i=0;i<numExp;i++)
		p.push_back(i);
	
	int t =0, ii=0;
	bool aux = true;
	
	maxerr = -1;
	double pmaxerr = -100;
	
	while(aux){
		if(t > ii * RANDPERIOD){
			random_shuffle(p.begin(),p.end());
			ii++;
		}
		for(int i=0; i < ceil(numExp/sizeBatch);i++){
			int aux2 =0;
			while(aux){
				aux = false;
				for(int x = sizeBatch*i;x < sizeBatch*(i+1) && x < numExp; x++){
					propagation(p[x]);
					backPropagation(p[x]);
					if(abs(out[x]-as[LAYERS][0]) > pmaxerr/FACTOR2)
						upWeight();
					if(abs(out[x]-as[LAYERS][0]) > 0.05 && abs(out[x] - as[LAYERS][0]) > pmaxerr/FACTOR2){
						aux=true;
					}
				}
				if(aux2 >= (int)BATCH/ceil(numExp/sizeBatch))
					break;
				aux2++;
				t++;
				if(maxerr != -1)
					pmaxerr = maxerr;
				else
					pmaxerr = -100;
				maxerr = -1;
			}
			aux2 = 0;
			while(aux){
				aux = false;	
				for(int x = 0; x < numExp; x++){
					propagation(x);
					backPropagation(x);
					if(abs(out[x]-as[LAYERS][0]) > pmaxerr/FACTOR)
						upWeight();
					if(abs(out[x] - as[LAYERS][0]) > 0.05 && abs(out[x]-as[LAYERS][0]) > pmaxerr/FACTOR){
						aux=true;
					}
				}
				if(aux2 >= (int)TODO/ceil(numExp/sizeBatch))
					break;
				aux2++;
				t++;
				if(maxerr != -1)
					pmaxerr = maxerr;
				else
					pmaxerr = -100;
				maxerr = -1;
			}
		}
		aux = false;
		for(int x =0; x < numExp;x++){
			propagation(x);
			backPropagation(x);
			if(abs(out[x]-as[LAYERS][0]) > pmaxerr/FACTOR)
						upWeight();
			if(abs(out[x] - as[LAYERS][0]) > 0.05 && abs(out[x]-as[LAYERS][0]) > pmaxerr/FACTOR){
				aux=true;
			}
		}
		if(maxerr != -1)
			pmaxerr = maxerr;
		else
			pmaxerr = -100;
		maxerr = -1;
		t++;
		if(t>=epo+1)
			break;
	}
	if(!aux){
		conv++;
		sum+= t;
		cout << "Repetitions " << k << ": " << t << endl;
		
		if(PRINT)
			printNet();
	}else{
		cout << "failed  " << k << endl;	
	}
}