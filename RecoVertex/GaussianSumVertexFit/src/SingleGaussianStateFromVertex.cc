#include "RecoVertex/GaussianSumVertexFit/interface/SingleGaussianStateFromVertex.h"

SingleGaussianStateFromVertex::SingleGaussianStateFromVertex(
			const VertexState & aState)
  : theState(aState)
{
  theWeight = aState.weightInMixture();
  GlobalPoint vertexPosition = aState.position();
  theMean = AlgebraicVector(3);
  theMean[0] = vertexPosition.x();
  theMean[1] = vertexPosition.y();
  theMean[2] = vertexPosition.z();
  theCovariance = aState.error().matrix();
}

ReferenceCountingPointer<SingleGaussianState>
SingleGaussianStateFromVertex::createNewState(const AlgebraicVector & aMean,
	const AlgebraicSymMatrix & aCovariance, double aWeight) const
{
  return RCSingleGaussianState(
   new SingleGaussianStateFromVertex(
  	VertexState(GlobalPoint(aMean[0], aMean[1], aMean[2]),
	GlobalError(aCovariance), aWeight) ) );
}
