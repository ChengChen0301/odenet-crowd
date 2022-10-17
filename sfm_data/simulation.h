#ifndef SIMULATION_H_INCLUDED
#define SIMULATION_H_INCLUDED

#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <stdlib.h>
#include <vector>
// #include <omp.h>
#include <limits>
#include <iomanip>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

void initialize();
double generateGaussianNoise(double mu, double sigma);
void computeForce();
void update(int t);
void savedata(int arg);


#endif // SIMULATION_H_INCLUDED
