#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexWeightCalculator.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

#include <cfloat>

double GsfVertexWeightCalculator::calculate(const  VertexState & oldVertex, //const  VertexState & newVertex,
	 const RefCountedLinearizedTrackState track, double cov) const
{
  double previousWeight = oldVertex.weightInMixture() * track->weightInMixture();
  // Jacobians
  const AlgebraicMatrixN3 & a = track->positionJacobian();
  const AlgebraicMatrixNM & b = track->momentumJacobian();

  //track information
  AlgebraicVectorN trackParameters = track->predictedStateParameters();
  AlgebraicSymMatrixNN trackParametersError = track->predictedStateError();
  //vertex information
  AlgebraicSymMatrix33 oldVertexError  = oldVertex.error().matrix_new();
  AlgebraicSymMatrix33 oldVertexWeight  = oldVertex.weight().matrix_new();
  //Vertex position
  GlobalPoint oldVertexPosition = oldVertex.position();
  AlgebraicVector3 oldVertexCoord;
  oldVertexCoord[0] = oldVertexPosition.x();
  oldVertexCoord[1] = oldVertexPosition.y();
  oldVertexCoord[2] = oldVertexPosition.z();

  // prior momentum information
  AlgebraicVector3 priorMomentum = track->predictedStateMomentumParameters();

  AlgebraicSymMatrix33 priorMomentumCov;
  priorMomentumCov(0,0) = 1.0E-9;
  priorMomentumCov(1,1) = 1.0E-6;
  priorMomentumCov(2,2) = 1.0E-6;
  priorMomentumCov *= cov;

  AlgebraicVectorN diff = trackParameters - track->constantTerm() -
    a*oldVertexCoord -b*priorMomentum;
  AlgebraicSymMatrixNN sigmaM = trackParametersError +
	ROOT::Math::Similarity(a,oldVertexError) +
	ROOT::Math::Similarity(b,priorMomentumCov);

  double sigmaDet;
  sigmaM.Det(sigmaDet);
  int  ifail = ! sigmaM.Invert(); 
  if(ifail != 0) throw VertexException
                       ("GsfVertexWeightCalculator::S matrix inversion failed");
  
	
  double chi = ROOT::Math::Similarity(diff,sigmaM);;  //SigmaM is now inverted !!!
  double weight = pow(2. * M_PI, -0.5 * 5) * sqrt(1./sigmaDet) * exp(-0.5 * chi);

 if (isnan(weight) || sigmaDet<=0.) {
   throw VertexException("GsfVertexWeightCalculator::Weight is nan");
  }

  return weight*previousWeight;

}
