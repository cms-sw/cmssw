#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"

#include <algorithm>

// Based on the R.Fruhwirth et al Computer Physics Communications 96 (1996) 189-208
CachingVertex KalmanVertexUpdator::update(const  CachingVertex & oldVertex,
	const RefCountedVertexTrack track, float weight, int sign) const
{
  if(abs(sign) != 1) throw VertexException
                          ("KalmanVertexUpdator::abs(sign) not equal to 1");

  VertexState newVertexState = positionUpdate(oldVertex.vertexState(), 
			  	track->linearizedTrack(), weight, sign);

  float chi1 = oldVertex.totalChiSquared();
  float chi2 = chi2Increment(oldVertex.vertexState(), newVertexState, 
                             track->linearizedTrack() , weight );
  chi1 +=sign * chi2;

//adding or removing track from the CachingVertex::VertexTracks
  vector<RefCountedVertexTrack> newVertexTracks = oldVertex.tracks();

  if (sign > 0) {
    newVertexTracks.push_back(track);
  }else{

    vector<RefCountedVertexTrack>::iterator pos 
      = find(newVertexTracks.begin(), newVertexTracks.end(), track);
    if (pos != newVertexTracks.end()) {
      newVertexTracks.erase(pos);
   }else{
     cout<<"KalmanVertexUpdator::Unable to find requested track in the current vertex"<<endl;
     throw VertexException("KalmanVertexUpdator::Unable to find requested track in the current vertex");
   }
  }
  if  (oldVertex.hasPrior()) {
    return CachingVertex( oldVertex.priorVertexState(),
                          newVertexState, newVertexTracks, chi1);
  } else {
    return CachingVertex(newVertexState, newVertexTracks, chi1);
  }
}


CachingVertex KalmanVertexUpdator::add(const CachingVertex & oldVertex, 
    const RefCountedVertexTrack track) const
{
  float weight = track->weight();
  return update(oldVertex,track,weight,+1);
}

CachingVertex KalmanVertexUpdator::remove(const CachingVertex & oldVertex, 
    const RefCountedVertexTrack track) const
{
  float weight = track->weight();
  return update(oldVertex,track,weight,-1);
}

float KalmanVertexUpdator::vertexPositionChi2( const VertexState& oldVertex,
	const GlobalPoint& newVertexPosition) const
{
  GlobalPoint oldVertexPosition = oldVertex.position();
  AlgebraicVector oldVertexPositionV(3);
  oldVertexPositionV[0] = oldVertexPosition.x();
  oldVertexPositionV[1] = oldVertexPosition.y();
  oldVertexPositionV[2] = oldVertexPosition.z();

  AlgebraicVector newVertexPositionV(3);
  newVertexPositionV[0] = newVertexPosition.x();
  newVertexPositionV[1] = newVertexPosition.y();
  newVertexPositionV[2] = newVertexPosition.z();

  AlgebraicVector positionResidual = newVertexPositionV - oldVertexPositionV;
  float result = oldVertex.weight().matrix().similarity(positionResidual);

  return result;
}


VertexState 
KalmanVertexUpdator::positionUpdate (const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 const float weight, int sign) const
{
  int ifail;
// Jacobians
  AlgebraicMatrix a = linearizedTrack->positionJacobian();
  AlgebraicMatrix b = linearizedTrack->momentumJacobian();

//track information
  AlgebraicVector trackParameters =
        linearizedTrack->predictedStateParameters();

  AlgebraicSymMatrix trackParametersWeight =
        linearizedTrack->predictedStateWeight();

//vertex information
  AlgebraicSymMatrix oldVertexWeight = oldVertex.weight().matrix();
  AlgebraicSymMatrix s = trackParametersWeight.similarityT(b);
  s.invert(ifail);
  if(ifail != 0) throw VertexException
                       ("KalmanVertexUpdator::S matrix inversion failed");

  AlgebraicSymMatrix gB = trackParametersWeight -
        s.similarity(b).similarity(trackParametersWeight);

// Getting the new covariance matrix of the vertex.

  AlgebraicSymMatrix newVertexWeight =  oldVertexWeight + weight * sign * gB.similarityT(a);

  AlgebraicVector newSwr =
                oldVertex.weightTimesPosition() + weight * sign * a.T() * gB *
                ( trackParameters - linearizedTrack->constantTerm());

  VertexState newpos (newSwr, GlobalWeight(newVertexWeight), 1.0);

  return newpos;
}


double KalmanVertexUpdator::chi2Increment(const VertexState & oldVertex, 
	const VertexState & newVertexState,
	const RefCountedLinearizedTrackState linearizedTrack, 
	float weight) const 
{
  GlobalPoint newVertexPosition = newVertexState.position();

  AlgebraicVector newVertexPositionV(3);
  newVertexPositionV[0] = newVertexPosition.x();
  newVertexPositionV[1] = newVertexPosition.y();
  newVertexPositionV[2] = newVertexPosition.z();

  AlgebraicMatrix a = linearizedTrack->positionJacobian();
  AlgebraicMatrix b = linearizedTrack->momentumJacobian();

//track information
  AlgebraicVector trackParameters =
  	linearizedTrack->predictedStateParameters();

  AlgebraicSymMatrix trackParametersWeight =
  	linearizedTrack->predictedStateWeight();

  int ifail;
  AlgebraicSymMatrix s = trackParametersWeight.similarityT(b);
  s.invert(ifail);
  if(ifail != 0) throw VertexException
                       ("KalmanVertexUpdator::S matrix inversion failed");

  AlgebraicVector theResidual = linearizedTrack->constantTerm();
  AlgebraicVector newTrackMomentumP =  s * b.T() * trackParametersWeight *
    (trackParameters - theResidual - a*newVertexPositionV);


  AlgebraicVector rtp = ( theResidual +  a * newVertexPositionV + b * newTrackMomentumP);

  AlgebraicVector parameterResiduals = trackParameters - rtp;

  double chi2 = weight * trackParametersWeight.similarity( parameterResiduals );

  chi2 += vertexPositionChi2(oldVertex, newVertexPosition);

  return chi2;
}
