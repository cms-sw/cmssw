#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

// Based on the R.Fruhwirth et al Computer Physics Communications 96 (1996) 189-208
template <unsigned int N>
CachingVertex<N> KalmanVertexUpdator<N>::update(const  CachingVertex<N> & oldVertex,
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

    typename vector<RefCountedVertexTrack>::iterator pos 
      = find(newVertexTracks.begin(), newVertexTracks.end(), track);
    if (pos != newVertexTracks.end()) {
      newVertexTracks.erase(pos);
   }else{
     cout<<"KalmanVertexUpdator::Unable to find requested track in the current vertex"<<endl;
     throw VertexException("KalmanVertexUpdator::Unable to find requested track in the current vertex");
   }
  }
  if  (oldVertex.hasPrior()) {
    return CachingVertex<N>( oldVertex.priorVertexState(),
                          newVertexState, newVertexTracks, chi1);
  } else {
    return CachingVertex<N>(newVertexState, newVertexTracks, chi1);
  }
}


template <unsigned int N>
CachingVertex<N> KalmanVertexUpdator<N>::add(const CachingVertex<N> & oldVertex, 
    const RefCountedVertexTrack track) const
{
  float weight = track->weight();
  return update(oldVertex,track,weight,+1);
}

template <unsigned int N>
CachingVertex<N> KalmanVertexUpdator<N>::remove(const CachingVertex<N> & oldVertex, 
    const RefCountedVertexTrack track) const
{
  float weight = track->weight();
  return update(oldVertex,track,weight,-1);
}

template <unsigned int N>
float KalmanVertexUpdator<N>::vertexPositionChi2( const VertexState& oldVertex,
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


template <unsigned int N>
VertexState 
KalmanVertexUpdator<N>::positionUpdate (const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linearizedTrack, 
	 const float weight, int sign) const
{
  int ifail;

  const AlgebraicMatrixN3 & a = linearizedTrack->positionJacobian();
  const AlgebraicMatrixNM & b = linearizedTrack->momentumJacobian();

//   AlgebraicVectorN trackParameters = 
//   	linearizedTrack->predictedStateParameters();

  AlgebraicSymMatrixNN trackParametersWeight = 
  	linearizedTrack->predictedStateWeight();


  // Jacobians
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "Now updating position" << "\n";

  //vertex information
//   AlgebraicSymMatrix33 oldVertexWeight = oldVertex.weight().matrix_new();
  AlgebraicSymMatrixMM s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  ifail = ! s.Invert(); 
  if(ifail != 0) throw VertexException
                       ("KalmanVertexUpdator::S matrix inversion failed");

  AlgebraicSymMatrixNN gB = trackParametersWeight -
       ROOT::Math::Similarity(trackParametersWeight, ROOT::Math::Similarity(b,s));

// Getting the new covariance matrix of the vertex.

  AlgebraicSymMatrix33 newVertexWeight =  oldVertex.weight().matrix_new()
	+ weight * sign * ROOT::Math::SimilarityT(a,gB);
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "weight matrix" << newVertexWeight << "\n";


  AlgebraicVector3 newSwr =
                oldVertex.weightTimesPosition() + weight * sign * ROOT::Math::Transpose(a) * gB *
                ( linearizedTrack->predictedStateParameters() - linearizedTrack->constantTerm());
  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "weighttimespos" << newSwr << "\n";

  VertexState newpos (newSwr, GlobalWeight(newVertexWeight), 1.0);

  //  edm::LogInfo("RecoVertex/KalmanVertexUpdator") 
  //    << "pos" << newpos.position() << "\n";

  return newpos;
}


template <unsigned int N>
double KalmanVertexUpdator<N>::chi2Increment(const VertexState & oldVertex, 
	const VertexState & newVertexState,
	const RefCountedLinearizedTrackState linearizedTrack, 
	float weight) const 
{
  GlobalPoint newVertexPosition = newVertexState.position();

  AlgebraicVector3 newVertexPositionV;
  newVertexPositionV(0) = newVertexPosition.x();
  newVertexPositionV(1) = newVertexPosition.y();
  newVertexPositionV(2) = newVertexPosition.z();

  const AlgebraicMatrixN3 & a = linearizedTrack->positionJacobian();
  const AlgebraicMatrixNM & b = linearizedTrack->momentumJacobian();

  AlgebraicVectorN trackParameters = 
  	linearizedTrack->predictedStateParameters();

  AlgebraicSymMatrixNN trackParametersWeight = 
  	linearizedTrack->predictedStateWeight();

  AlgebraicSymMatrixMM s = ROOT::Math::SimilarityT(b,trackParametersWeight);
  bool ret = s.Invert(); 
  if(!ret) throw VertexException
                       ("KalmanVertexUpdator::S matrix inversion failed");

  const AlgebraicVectorN & theResidual = linearizedTrack->constantTerm();
  AlgebraicVectorM newTrackMomentumP =  s * ROOT::Math::Transpose(b) * trackParametersWeight *
    (trackParameters - theResidual - a*newVertexPositionV);


  AlgebraicVectorN rtp = ( theResidual +  a * newVertexPositionV + b * newTrackMomentumP);

  AlgebraicVectorN parameterResiduals = trackParameters - rtp;
  if (parameterResiduals(2) >  M_PI) parameterResiduals(2)-= 2*M_PI;
  if (parameterResiduals(2) < -M_PI) parameterResiduals(2)+= 2*M_PI;

  double chi2 = weight * ROOT::Math::Similarity(parameterResiduals, trackParametersWeight);

//   chi2 += vertexPositionChi2(oldVertex, newVertexPosition);
  chi2 += helper.vertexChi2(oldVertex, newVertexState);

  return chi2;
}

template class KalmanVertexUpdator<5>;
template class KalmanVertexUpdator<6>;
