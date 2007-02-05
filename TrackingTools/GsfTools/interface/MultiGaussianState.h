#ifndef MultiGaussianState_H
#define MultiGaussianState_H

#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"
#include <vector>

/**
 * Base Class for a mixture of multi-variate gaussian state.
 */

class MultiGaussianState : public ReferenceCounted {

public:

  MultiGaussianState() : theCombinedStateUp2Date(false) {}

  MultiGaussianState(const std::vector<RCSingleGaussianState> & stateV)
    : theComponents(stateV), theCombinedStateUp2Date(false) {}

  /**
   * Creates a new multi-state with the given components.
   * For this base class, no information is passed from the initial
   * instance.
   */
  virtual ReferenceCountingPointer<MultiGaussianState> createNewState(
	const std::vector<RCSingleGaussianState> & stateV) const {
    return ReferenceCountingPointer<MultiGaussianState>(
      new MultiGaussianState(stateV) );
  }

  /**
   * Creates a new single-state with the given information.
   * For this base class, no information is passed from the initial
   * instance.
   */
  virtual RCSingleGaussianState createSingleState (
	const AlgebraicVector & aMean, const AlgebraicSymMatrix & aCovariance,
	double aWeight = 1.) const {
    return ReferenceCountingPointer<SingleGaussianState>(
      new SingleGaussianState(aMean, aCovariance, aWeight) );
  }

  double weight() const;
  const AlgebraicVector & mean() const;
  const AlgebraicSymMatrix & covariance() const;
  const std::vector<RCSingleGaussianState> & components() {return theComponents;}

protected:

  void checkCombinedState() const;

  std::vector<RCSingleGaussianState> theComponents;
  mutable RCSingleGaussianState theCombinedState;
  mutable bool theCombinedStateUp2Date;
};

#endif
