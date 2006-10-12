#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include <algorithm>

struct vT_find
{
  bool operator()(const CachingVertex & v, const RefCountedVertexTrack t)
  {
//initial tracks  
    vector<RefCountedVertexTrack> tracks = v.tracks();
    vector<RefCountedVertexTrack>::iterator pos 
      = find(tracks.begin(), tracks.end(), t);
    return (pos != tracks.end());
  }
}; 
 
 
float 
KalmanVertexTrackCompatibilityEstimator::estimate(const CachingVertex & vertex,
			 const RefCountedVertexTrack tr) const
{
//checking if the track passed really belongs to the vertex
 vT_find finder;
 if(finder(vertex,tr)) {
   return estimateFittedTrack(vertex,tr);
 } else {
   return estimateNFittedTrack(vertex,tr);
 }
} 


float
KalmanVertexTrackCompatibilityEstimator::estimate(const CachingVertex & vertex, 
			 const RefCountedLinearizedTrackState track) const
{
  RefCountedVertexTrack vertexTrack = vTrackFactory.vertexTrack(track,
 						 vertex.vertexState());
  return estimate(vertex, vertexTrack);
}


float
KalmanVertexTrackCompatibilityEstimator::estimate(const reco::Vertex & vertex, 
			 const reco::TransientTrack & track) const
{ 	
//   GlobalPoint linP(vertex.position().x(), vertex.position().z(),vertex.position().z());
    GlobalPoint linP(Basic3DVector<float> (vertex.position()));

  RefCountedLinearizedTrackState linTrack = 
  			lTrackFactory.linearizedTrackState(linP, track);
  GlobalError err(RecoVertex::convertError(vertex.covariance()));
  VertexState vState(linP, err);
  RefCountedVertexTrack vertexTrack = vTrackFactory.vertexTrack(linTrack, vState);

  vector<RefCountedVertexTrack> initialTracks(1, vertexTrack);
  CachingVertex cachingVertex(linP, err, initialTracks,
  			    vertex.chi2());
  // FIXME: this should work also for tracks without a persistent ref.
  if (find(vertex.tracks_begin(), vertex.tracks_end(), track.persistentTrackRef()) != vertex.tracks_end())
  {
    return estimateFittedTrack(cachingVertex, vertexTrack);
  } else {
    return estimateNFittedTrack(cachingVertex, vertexTrack);
  }
}



// methods to calculate track<-->vertex compatibility
// with the track belonging to the vertex

float 
KalmanVertexTrackCompatibilityEstimator::estimateFittedTrack
		(const CachingVertex & v, const RefCountedVertexTrack track) const
{
  //remove track from the vertex using the vertex updator
  // Using the update instead of the remove methode, we can specify a weight which
  // is different than then one which the vertex track has been defined with.
  CachingVertex rVert = updator.remove(v, track);
  RefCountedVertexTrack newSmoothedTrack = trackUpdator.update(v, track);
  return estimateDifference(v,rVert,newSmoothedTrack);
}

// method calculating track<-->vertex compatibility
//with the track not belonging to vertex
float KalmanVertexTrackCompatibilityEstimator::estimateNFittedTrack
 	(const CachingVertex & v, const RefCountedVertexTrack track) const
{
  // Using the update instead of the add methode, we can specify a weight which
  // is different than then one which the vertex track has been defined with.
  CachingVertex rVert = updator.add(v, track);
  return (rVert.totalChiSquared()-v.totalChiSquared());
}   



float KalmanVertexTrackCompatibilityEstimator::estimateDifference
	(const CachingVertex & more, const CachingVertex & less, 
         const RefCountedVertexTrack track) const
{
//initial vertex position
 GlobalPoint iPos = more.position();
 AlgebraicVector iV(3);
 iV[0] = iPos.x();
 iV[1] = iPos.y();
 iV[2] = iPos.z();

//position and covariance matrix after removing 
 GlobalPoint rPos = less.position();
 AlgebraicSymMatrix nWeight = less.weight().matrix();
 AlgebraicVector rV(3);
 rV[0] = rPos.x();
 rV[1] = rPos.y();
 rV[2] = rPos.z();

//smoothed residuals
 AlgebraicVector parameterResiduals = 
	  track->linearizedTrack()->predictedStateParameters() -
//	  track->linearizedTrack()->predictedState().perigeeParameters().vector() - 
	  track->refittedParamFromEquation();
//position residuals
 AlgebraicVector posResiduals = iV - rV;	
   
//track parameters weight
  AlgebraicSymMatrix w = track->linearizedTrack()->predictedStateWeight();
  //track->linearizedTrack()->predictedState().perigeeError().weightMatrix();  
   
//chi2   
 return(nWeight.similarity(posResiduals) + track->weight()*w.similarity(parameterResiduals));

}

