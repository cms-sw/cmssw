#include "RecoVertex/AdaptiveVertexFit/interface/KalmanChiSquare.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

using namespace std;

float KalmanChiSquare::estimate(
      const GlobalPoint & vertex,
      RefCountedLinearizedTrackState lt ) const
{
  AlgebraicVector3 vtxposV;
  vtxposV[0] = vertex.x();
  vtxposV[1] = vertex.y();
  vtxposV[2] = vertex.z();

  AlgebraicMatrix53 a = lt->positionJacobian();
  AlgebraicMatrix53 b = lt->momentumJacobian();

  // track information
  AlgebraicVector5 trackParameters =
      lt->predictedStateParameters();

  AlgebraicSymMatrix55 trackParametersWeight =
      lt->predictedStateWeight();

  AlgebraicSymMatrix33 s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  bool ret = s.Invert(); 
  if(!ret) throw VertexException
                       ("[KalmanChiSquare] S matrix inversion failed");

  AlgebraicVector5 residual = lt->constantTerm();
  AlgebraicVector3 newTrackMomentumP =  s * ROOT::Math::Transpose(b) * trackParametersWeight *
     (trackParameters - residual - a*vtxposV);

  AlgebraicVector5 rtp = ( residual +  a * vtxposV + b * newTrackMomentumP);

  AlgebraicVector5 parameterResiduals = trackParameters - rtp;
  lt->checkParameters(parameterResiduals);

  return ROOT::Math::Similarity(parameterResiduals, trackParametersWeight);
}
