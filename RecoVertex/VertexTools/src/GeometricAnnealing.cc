#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include <cmath>
#include <iostream>
#include <limits>

GeometricAnnealing::GeometricAnnealing (
     const double cutoff, const double T, const double ratio ) :
  theT0(T), theT(T), theChi2cut(cutoff*cutoff), theRatio( ratio )
{}

void GeometricAnnealing::anneal()
{
  theT=1+(theT-1)*theRatio;
}

double GeometricAnnealing::weight ( double chi2 ) const
{
  double mphi = phi ( chi2 );
  long double newtmp = mphi / ( mphi + phi ( theChi2cut ) );
  if ( edm::isNotFinite(newtmp) )
  {
    if ( chi2 < theChi2cut ) newtmp=1.;
    else newtmp=0.;
  }
  return newtmp;
}

void GeometricAnnealing::resetAnnealing()
{
  theT=theT0;
}

double GeometricAnnealing::phi( double chi2 ) const
{
  return exp ( -.5 * chi2 / theT );
}

double GeometricAnnealing::cutoff() const
{
  // std::cout << "[GeometricAnnealing] cutoff called!" << std::endl;
  return sqrt(theChi2cut);
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
  std::cout << "[GeometricAnnealing] chi2_cut=" << theChi2cut << ", Tini="
       << theT0 << ", ratio=" << theRatio << std::endl;
}
