#ifndef MultiGaussianStateFromVertex_H
#define MultiGaussianStateFromVertex_H

#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState.h"
#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"

/**
 * Mixture of multi-variate gaussian state constructed from a VertexState
 */

class MultiGaussianStateFromVertex : public MultiGaussianState {

public:

  MultiGaussianStateFromVertex(const VertexState & aState);

  /**
   * Creates a new multi-state with the given components.
   * For this class, no information is passed from the initial VertexState.
   */
  virtual ReferenceCountingPointer<MultiGaussianState> createNewState(
	const std::vector<RCSingleGaussianState> & stateV) const;

  /**
   * Creates a new single-state with the given information.
   * For this class, no information is passed from the initial VertexState.
   */
  virtual RCSingleGaussianState createSingleState (
	const AlgebraicVector & aMean, const AlgebraicSymMatrix & aCovariance,
	double aWeight = 1.) const;

  /**
   * The VertexState the object was constructed with.
   */
  VertexState state() const {return theState;}

private:

  VertexState theState;
};

#endif
