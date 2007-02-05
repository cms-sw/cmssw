#ifndef RCGaussianStateFactory_H
#define RCGaussianStateFactory_H

#include "RecoVertex/GaussianSumVertexFit/interface/SingleGaussianStateFromVertex.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiGaussianStateFromVertex.h"
#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/RCMultiGaussianState.h"

/**
 *  Concrete class to encapsulate the creation of Reference counted gaussian state
 *  Should always be used in order to create a new RefCounted gaussian states,
 *  so that the reference-counting mechanism works well.
 */

class RCGaussianStateFactory {

public:

//   RCSingleGaussianState singleGaussianState(const RCSingleGaussianState & aState) const;

  RCSingleGaussianState singleGaussianState(const VertexState & aState) const;

  RCSingleGaussianState singleGaussianState(const AlgebraicVector & aMean,
	const AlgebraicSymMatrix & aCovariance, double aWeight = 1.) const;

  RCMultiGaussianState multiGaussianState(const VertexState & aState) const;


};

#endif
