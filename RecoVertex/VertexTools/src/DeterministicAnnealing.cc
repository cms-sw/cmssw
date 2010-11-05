#include "RecoVertex/VertexTools/interface/DeterministicAnnealing.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;

namespace {
  vector < float > temperatures;
}

DeterministicAnnealing::DeterministicAnnealing ( float cutoff ) :
  theIndex(0), theChi2cut ( cutoff*cutoff ), theIsAnnealed ( false )
{
  temperatures.push_back(256);
  temperatures.push_back(64);
  temperatures.push_back(16);
  temperatures.push_back(4);
  temperatures.push_back(2);
  temperatures.push_back(1);
}

DeterministicAnnealing::DeterministicAnnealing( const vector < float > & sched,
     float cutoff ) : theIndex(0), theChi2cut ( cutoff*cutoff ), theIsAnnealed ( false )
{
  temperatures = sched;
}

void DeterministicAnnealing::anneal()
{
  if ( theIndex < ( temperatures.size() - 1 ) )
  {
    theIndex++; 
  } else {
    theIsAnnealed = true;
  };
}

double DeterministicAnnealing::weight ( double chi2 ) const
{
  long double mphi = phi ( chi2 );
  /*
  if ( mphi < std::numeric_limits<double>::epsilon() ) return 0.;
  return 1. / ( 1. + phi ( theChi2cut * theChi2cut ) / mphi );
  */
  // return mphi / ( mphi + phi ( theChi2cut ) );
  long double newtmp = mphi / ( mphi + phi ( theChi2cut ) );
  if ( std::isinf(newtmp ) )
  {
    if ( chi2 < theChi2cut ) newtmp=1.;
    else newtmp=0.;
  }
  return newtmp;
}

void DeterministicAnnealing::resetAnnealing()
{
  theIndex=0;
  theIsAnnealed = false;
}

inline double DeterministicAnnealing::phi( double chi2 ) const
{
  return exp ( -.5 * chi2 / temperatures[theIndex] );
}

double DeterministicAnnealing::cutoff() const
{
  return sqrt(theChi2cut);
}

double DeterministicAnnealing::currentTemp() const
{
  return temperatures[theIndex];
}

double DeterministicAnnealing::initialTemp() const
{
  return temperatures[0];
}

bool DeterministicAnnealing::isAnnealed() const
{
  return theIsAnnealed;
}

void DeterministicAnnealing::debug() const
{
  cout << "[DeterministicAnnealing] schedule=";
  for ( vector< float >::const_iterator i=temperatures.begin(); 
        i!=temperatures.end() ; ++i )
  {
    cout << *i << " ";
  };
  cout << endl;
}
