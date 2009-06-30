#ifndef SingleGaussianState1D_H
#define SingleGaussianState1D_H

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "boost/shared_ptr.hpp"

/** One-dimensional (single) Gaussian state. Used for the description 
 *  of Gaussian mixtures in one dimension.
 */

class SingleGaussianState1D {
// private:
//   typedef SingleGaussianState<1> SingleState;
  
public:
  
  SingleGaussianState1D () :
    theWeight(0.), theMean(0.), theVariance(0.), theStandardDeviation(-1.) {}
  
  SingleGaussianState1D (double aMean,
			 double aVariance, 
			 double aWeight = 1.) : 
    theWeight(aWeight), theMean(aMean), theVariance(aVariance), theStandardDeviation(-1.) {}
  
  ~SingleGaussianState1D () {}
  
  /// weight
  inline double weight() const {return theWeight;}
  /// parameter vector
  inline double mean() const {return theMean;}
  /// variance
  inline double variance() const {return theVariance;}
  /// standardDeviation
  double standardDeviation() const {
    if ( theStandardDeviation<0. )  theStandardDeviation = sqrt(theVariance);
    return theStandardDeviation;
  }
//   /// state
//   boost::shared_ptr<SingleState> state() {return theState;}

private:
  double theWeight;
  double theMean;
  double theVariance;
  mutable double theStandardDeviation;
};

#endif
