#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include <cmath>
#include <iostream>
#include <limits>

GeometricAnnealing::GeometricAnnealing (
     const double cutoff, const double T, const double ratio ) :
  theT0(T), theT(T), theCutoff(cutoff), theRatio( ratio )
{}

void GeometricAnnealing::anneal()
{
  theT=1+(theT-1)*theRatio;
}

double GeometricAnnealing::weight ( double chi2 ) const
{
  double mphi = phi ( chi2 );
  if ( mphi < std::numeric_limits<double>::epsilon() ) return 0.;
  return 1. / ( 1. + phi ( theCutoff * theCutoff ) / mphi );
}

void GeometricAnnealing::resetAnnealing()
{
  theT=theT0;
}

inline double GeometricAnnealing::phi( double chi2 ) const
{
  return exp ( -.5 * chi2 / theT );
}

double GeometricAnnealing::cutoff() const
{
  return theCutoff;
}

double GeometricAnnealing::currentTemp() const
{
  return theT;
}

double GeometricAnnealing::initialTemp() const
{
  return theT0;
}

bool GeometricAnnealing::isAnnealed() const
{
  return ( theT < 1.02 );
}

void GeometricAnnealing::debug() const
{
  std::cout << "[GeometricAnnealing] sigma_cut=" << theCutoff << ", Tini="
       << theT0 << ", ratio=" << theRatio << std::endl;
}
