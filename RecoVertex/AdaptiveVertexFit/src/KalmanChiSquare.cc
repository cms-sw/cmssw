#include "RecoVertex/AdaptiveVertexFit/interface/KalmanChiSquare.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

using namespace std;

float KalmanChiSquare::estimate(
      const GlobalPoint & vertex,
      RefCountedLinearizedTrackState lt ) const
{
  AlgebraicVector vtxposV(3);
  vtxposV[0] = vertex.x();
  vtxposV[1] = vertex.y();
  vtxposV[2] = vertex.z();

  AlgebraicMatrix a = lt->positionJacobian();
  AlgebraicMatrix b = lt->momentumJacobian();

  // track information
  AlgebraicVector trackParameters =
      lt->predictedStateParameters();

  AlgebraicSymMatrix trackParametersWeight =
      lt->predictedStateWeight();

  int ifail;
  AlgebraicSymMatrix s = trackParametersWeight.similarityT(b);
  s.invert(ifail);
  if(ifail != 0) throw VertexException
                       ("[KalmanChiSquare] S matrix inversion failed");

  AlgebraicVector residual = lt->constantTerm();
  AlgebraicVector newTrackMomentumP =  s * b.T() * trackParametersWeight *
    (trackParameters - residual - a*vtxposV);

  AlgebraicVector rtp = ( residual +  a * vtxposV + b * newTrackMomentumP);

  AlgebraicVector parameterResiduals = trackParameters - rtp;

  return trackParametersWeight.similarity( parameterResiduals );
}
