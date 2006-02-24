#include "RecoVertex/VertexTools/interface/DeterministicAnnealing.h"
#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

namespace {
  vector < float > temperatures;
};

DeterministicAnnealing::DeterministicAnnealing ( float cutoff ) :
  theIndex(0), theCutoff ( cutoff ), theIsAnnealed ( false )
{
  temperatures.push_back(256);
  temperatures.push_back(64);
  temperatures.push_back(16);
  temperatures.push_back(4);
  temperatures.push_back(2);
  temperatures.push_back(1);
};

DeterministicAnnealing::DeterministicAnnealing( const vector < float > & sched,
     float cutoff ) : theIndex(0), theCutoff ( cutoff ), theIsAnnealed ( false )
{
  temperatures = sched;
};

void DeterministicAnnealing::anneal()
{
  if ( theIndex < ( temperatures.size() - 1 ) )
  {
    theIndex++; 
  } else {
    theIsAnnealed = true;
  };
};

double DeterministicAnnealing::weight ( double chi2 ) const
{
  return 1. / ( 1. + phi ( theCutoff * theCutoff ) / phi ( chi2 ) );
};

void DeterministicAnnealing::resetAnnealing()
{
  theIndex=0;
  theIsAnnealed = false;
};

inline double DeterministicAnnealing::phi( double chi2 ) const
{
  return exp ( -.5 * chi2 / temperatures[theIndex] );
};

double DeterministicAnnealing::cutoff() const
{
  return theCutoff;
};

double DeterministicAnnealing::currentTemp() const
{
  return temperatures[theIndex];
};

double DeterministicAnnealing::initialTemp() const
{
  return temperatures[0];
};

bool DeterministicAnnealing::isAnnealed() const
{
  return theIsAnnealed;
};

void DeterministicAnnealing::debug() const
{
  cout << "[DeterministicAnnealing] schedule=";
  for ( vector< float >::const_iterator i=temperatures.begin(); 
        i!=temperatures.end() ; ++i )
  {
    cout << *i << " ";
  };
  cout << endl;
};
