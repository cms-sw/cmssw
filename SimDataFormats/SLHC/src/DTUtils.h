#ifndef __UUTILS_
#define __UUTILS_


#include <vector> 

static const double pi_greca = 3.14159265358979323846;

double deltaPhi(double ph1, double phi2);
float  deltaPhi(float phi1, float phi2);
int    deltaPhi(int ph1, int phi2);

// to write in binary format
char* biny (int Value, bool truncated_on_the_left = false , size_t wordSize = 32);

#endif

