#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexWeightCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <cfloat>

double GsfVertexWeightCalculator::calculate(const  VertexState & oldVertex, //const  VertexState & newVertex,
	 const RefCountedLinearizedTrackState track, double cov) const
{
  double previousWeight = oldVertex.weightInMixture() * track->weightInMixture();
  // Jacobians
  AlgebraicMatrix a = track->positionJacobian();
  AlgebraicMatrix b = track->momentumJacobian();

  //track information
  AlgebraicVector trackParameters = track->predictedStateParameters();
  AlgebraicSymMatrix trackParametersError = track->predictedStateError();
  //vertex information
  AlgebraicSymMatrix oldVertexError  = oldVertex.error().matrix();
  AlgebraicSymMatrix oldVertexWeight  = oldVertex.weight().matrix();
  //Vertex position
  GlobalPoint oldVertexPosition = oldVertex.position();
  AlgebraicVector oldVertexCoord(3);
  oldVertexCoord[0] = oldVertexPosition.x();
  oldVertexCoord[1] = oldVertexPosition.y();
  oldVertexCoord[2] = oldVertexPosition.z();

  // prior momentum information
  AlgebraicVector priorMomentum = track->predictedStateMomentumParameters();

  AlgebraicSymMatrix priorMomentumCov(3,0);
  priorMomentumCov(1,1) = 1.0E-9;
  priorMomentumCov(2,2) = 1.0E-6;
  priorMomentumCov(3,3) = 1.0E-6;
  priorMomentumCov *= cov;

  AlgebraicVector diff = trackParameters - track->constantTerm() -
    a*oldVertexCoord -b*priorMomentum;
  AlgebraicSymMatrix sigmaM = trackParametersError +
    oldVertexError.similarity(a) + priorMomentumCov.similarity(b);
// cout << "Condition: "<< condition(sigmaM)<<endl;
  double sigmaDet = sigmaM.determinant();
  int ifail;
  AlgebraicSymMatrix sigmaMInv = sigmaM.inverse(ifail);
  double chi = sigmaMInv.similarity(diff);
  double weight = pow(2. * M_PI, -0.5 * 5) * sqrt(1./sigmaDet) * exp(-0.5 * chi);

 if (isnan(weight) || sigmaDet<=0.) {
   throw VertexException("GsfVertexWeightCalculator::Weight is nan");
  }

  return weight*previousWeight;

}
