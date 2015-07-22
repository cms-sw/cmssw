#include "RecoVertex/VertexTools/interface/DeterministicAnnealing.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;

DeterministicAnnealing::DeterministicAnnealing ( float cutoff ) :
  theTemperatures({256,64,16,4,2,1}),
  theIndex(0), theChi2cut ( cutoff*cutoff ), theIsAnnealed ( false )
{
}

DeterministicAnnealing::DeterministicAnnealing( const vector < float > & sched, float cutoff ) :
  theTemperatures(sched),theIndex(0), theChi2cut ( cutoff*cutoff ), theIsAnnealed ( false )
{
}

void DeterministicAnnealing::anneal()
{
  if ( theIndex < ( theTemperatures.size() - 1 ) )
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
  if ( edm::isNotFinite(newtmp ) )
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
  return exp ( -.5 * chi2 / theTemperatures[theIndex] );
}

double DeterministicAnnealing::cutoff() const
{
  return sqrt(theChi2cut);
}

double DeterministicAnnealing::currentTemp() const
{
  return theTemperatures[theIndex];
}

double DeterministicAnnealing::initialTemp() const
{
  return theTemperatures[0];
}

bool DeterministicAnnealing::isAnnealed() const
{
  return theIsAnnealed;
}

void DeterministicAnnealing::debug() const
{
  cout << "[DeterministicAnnealing] schedule=";
  for ( vector< float >::const_iterator i=theTemperatures.begin(); 
        i!=theTemperatures.end() ; ++i )
  {
    cout << *i << " ";
  };
  cout << endl;
}
