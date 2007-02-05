#include "RecoVertex/GaussianSumVertexFit/interface/RCGaussianStateFactory.h"

// RCSingleGaussianState
// RCGaussianStateFactory::singleGaussianState(const RCSingleGaussianState & aState) const
// {
//   return RCSingleGaussianState( new SingleGaussianState(aState) );
// }

RCSingleGaussianState
RCGaussianStateFactory::singleGaussianState(const VertexState & aState) const
{
  return RCSingleGaussianState( new SingleGaussianStateFromVertex(aState) );
}

RCSingleGaussianState
RCGaussianStateFactory::singleGaussianState(const AlgebraicVector & aMean,
	const AlgebraicSymMatrix & aCovariance, double aWeight) const
{
  return RCSingleGaussianState( new SingleGaussianState(aMean,
  	aCovariance, aWeight) );
}

RCMultiGaussianState
RCGaussianStateFactory::multiGaussianState(const VertexState & aState) const
{
  return RCMultiGaussianState( new MultiGaussianStateFromVertex(aState) );
}
