#ifndef SingleGaussianState_H
#define SingleGaussianState_H

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/**
 * Base Class for a single multi-variate gaussian state.
 */

class SingleGaussianState  : public ReferenceCounted {

public:

  SingleGaussianState() {}

  SingleGaussianState(const AlgebraicVector & aMean,
	 const AlgebraicSymMatrix & aCovariance, double aWeight = 1.)
    : theWeight(aWeight), theMean(aMean), theCovariance(aCovariance) {}

  virtual ~SingleGaussianState() {}

  /**
   * Creates a new single-state with the given information.
   * For this base class, no information is passed from the initial 
   * instance.
   */
  virtual ReferenceCountingPointer<SingleGaussianState> 
  	createNewState(const AlgebraicVector & aMean, 
	  const AlgebraicSymMatrix & aCovariance, double aWeight = 1) const {
    return ReferenceCountingPointer<SingleGaussianState>(
      new SingleGaussianState(aMean, aCovariance, aWeight) );
  }

  double weight() const {return theWeight;}
  const AlgebraicVector & mean() const {return theMean;}
  const AlgebraicSymMatrix & covariance() const {return theCovariance;}

protected:

  double theWeight;
  AlgebraicVector theMean;
  AlgebraicSymMatrix theCovariance;
};

#endif
