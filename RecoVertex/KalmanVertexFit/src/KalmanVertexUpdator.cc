#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  AlgebraicVector3 oldVertexPositionV;
  oldVertexPositionV(0) = oldVertexPosition.x();
  oldVertexPositionV(1) = oldVertexPosition.y();
  oldVertexPositionV(2) = oldVertexPosition.z();

  AlgebraicVector3 newVertexPositionV;
  newVertexPositionV(0) = newVertexPosition.x();
  newVertexPositionV(1) = newVertexPosition.y();
  newVertexPositionV(2) = newVertexPosition.z();

  AlgebraicVector3 positionResidual = newVertexPositionV - oldVertexPositionV;
  float result = ROOT::Math::Similarity(positionResidual, oldVertex.weight().matrix_new());

  return result;
}


VertexState 
KalmanVertexUpdator::positionUpdate (const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 const float weight, int sign) const
{
  int ifail;
  // Jacobians
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "Now updating position" << "\n";
  AlgebraicMatrix53 a = linearizedTrack->positionJacobian();
  AlgebraicMatrix53 b = linearizedTrack->momentumJacobian();
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "got jacobians" << "\n";
  
  //track information
  AlgebraicVector5 trackParameters =
        linearizedTrack->predictedStateParameters();

  AlgebraicSymMatrix55 trackParametersWeight =
        linearizedTrack->predictedStateWeight();

  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "got track parameters" << "\n";

  //vertex information
  AlgebraicSymMatrix33 oldVertexWeight = oldVertex.weight().matrix_new();
  AlgebraicSymMatrix33 s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  ifail = ! s.Invert(); 
  if(ifail != 0) throw VertexException
                       ("KalmanVertexUpdator::S matrix inversion failed");

  AlgebraicSymMatrix55 gB = trackParametersWeight -
       ROOT::Math::Similarity(trackParametersWeight, ROOT::Math::Similarity(b,s));

// Getting the new covariance matrix of the vertex.

  AlgebraicSymMatrix33 newVertexWeight =  oldVertexWeight + weight * sign * ROOT::Math::SimilarityT(a,gB);
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "weight matrix" << newVertexWeight << "\n";


  AlgebraicVector3 newSwr =
                oldVertex.weightTimesPosition() + weight * sign * ROOT::Math::Transpose(a) * gB *
                ( trackParameters - linearizedTrack->constantTerm());
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "weighttimespos" << newSwr << "\n";

  VertexState newpos (newSwr, GlobalWeight(newVertexWeight), 1.0);

  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "pos" << newpos.position() << "\n";

  return newpos;
}


double KalmanVertexUpdator::chi2Increment(const VertexState & oldVertex, 
	const VertexState & newVertexState,
	const RefCountedLinearizedTrackState linearizedTrack, 
	float weight) const 
{
  GlobalPoint newVertexPosition = newVertexState.position();

  AlgebraicVector3 newVertexPositionV;
  newVertexPositionV(0) = newVertexPosition.x();
  newVertexPositionV(1) = newVertexPosition.y();
  newVertexPositionV(2) = newVertexPosition.z();

  AlgebraicMatrix53 a = linearizedTrack->positionJacobian();
  AlgebraicMatrix53 b = linearizedTrack->momentumJacobian();

//track information
  AlgebraicVector5 trackParameters =
  	linearizedTrack->predictedStateParameters();

  AlgebraicSymMatrix55 trackParametersWeight =
  	linearizedTrack->predictedStateWeight();

  AlgebraicSymMatrix33 s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  bool ret = s.Invert(); 
  if(!ret) throw VertexException
                       ("KalmanVertexUpdator::S matrix inversion failed");

  AlgebraicVector5 theResidual = linearizedTrack->constantTerm();
  AlgebraicVector3 newTrackMomentumP =  s * ROOT::Math::Transpose(b) * trackParametersWeight *
    (trackParameters - theResidual - a*newVertexPositionV);


  AlgebraicVector5 rtp = ( theResidual +  a * newVertexPositionV + b * newTrackMomentumP);

  AlgebraicVector5 parameterResiduals = trackParameters - rtp;

  double chi2 = weight * ROOT::Math::Similarity(parameterResiduals, trackParametersWeight);

//   chi2 += vertexPositionChi2(oldVertex, newVertexPosition);
  chi2 += helper.vertexChi2(oldVertex, newVertexState);

  return chi2;
}
