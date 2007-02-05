#ifndef SingleGaussianStateFromVertex_H
#define SingleGaussianStateFromVertex_H

#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"

/**
 * Single multi-variate gaussian state constructed from a VertexState
 */

class SingleGaussianStateFromVertex : public SingleGaussianState {

public:

  SingleGaussianStateFromVertex(const VertexState & aState);

  /**
   * Creates a new single-state with the given information.
   * For this class, no information is passed from the initial VertexState.
   */
  virtual ReferenceCountingPointer<SingleGaussianState>
    createNewState(const AlgebraicVector & aMean,
	  const AlgebraicSymMatrix & aCovariance, double aWeight = 1.) const;

  /**
   * The VertexState the object was constructed with.
   */
  VertexState state() const {return theState;}

private:

  VertexState theState;
};

#endif
