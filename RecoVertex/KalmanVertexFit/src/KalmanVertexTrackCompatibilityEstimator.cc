#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include <algorithm>
using namespace reco;


template <unsigned int N>
typename KalmanVertexTrackCompatibilityEstimator<N>::BDpair
KalmanVertexTrackCompatibilityEstimator<N>::estimate(const CachingVertex<N> & vertex,
						     const RefCountedVertexTrack tr,
						     unsigned int hint) const
{
//checking if the track passed really belongs to the vertex

  const std::vector<RefCountedVertexTrack> &tracks = vertex.tracksRef();

  if ( tracks.size()==0)
   return estimateNFittedTrack(vertex,tr);

  if (hint<tracks.size() ) {
    VertexTrackEqual<N> d(tr);
    if ( d(tracks[hint]))
      return estimateFittedTrack(vertex,tracks[hint]);
  }

  typename std::vector<RefCountedVertexTrack>::const_iterator pos 
    = find_if(tracks.begin(), tracks.end(), VertexTrackEqual<N>(tr));
  if(pos != tracks.end()) {
    return estimateFittedTrack(vertex,*pos);
  } else {
    return estimateNFittedTrack(vertex,tr);
  }
} 


template <unsigned int N>
typename KalmanVertexTrackCompatibilityEstimator<N>::BDpair
KalmanVertexTrackCompatibilityEstimator<N>::estimate(const CachingVertex<N> & vertex, 
						     const RefCountedLinearizedTrackState track,
						     unsigned int hint) const
{
  RefCountedVertexTrack vertexTrack = vTrackFactory.vertexTrack(track,
 						 vertex.vertexState());
  return estimate(vertex, vertexTrack,hint);
}


template <unsigned int N>
typename KalmanVertexTrackCompatibilityEstimator<N>::BDpair
KalmanVertexTrackCompatibilityEstimator<N>::estimate(const reco::Vertex & vertex, 
			 const reco::TransientTrack & track) const
{ 	
//   GlobalPoint linP(vertex.position().x(), vertex.position().z(),vertex.position().z());
    GlobalPoint linP(Basic3DVector<float> (vertex.position()));

  LinearizedTrackStateFactory lTrackFactory;
  RefCountedLinearizedTrackState linTrack = 
  			lTrackFactory.linearizedTrackState(linP, track);
  GlobalError err(vertex.covariance());
  VertexState vState(linP, err);
  RefCountedVertexTrack vertexTrack = vTrackFactory.vertexTrack(linTrack, vState);

  std::vector<RefCountedVertexTrack> initialTracks(1, vertexTrack);
  CachingVertex<N> cachingVertex(linP, err, initialTracks,
  			    vertex.chi2());
  // FIXME: this should work also for tracks without a persistent ref.
//   return estimateNFittedTrack(cachingVertex, vertexTrack);
  if (find(vertex.tracks_begin(), vertex.tracks_end(), track.trackBaseRef()) != vertex.tracks_end())
  {
    return estimateFittedTrack(cachingVertex, vertexTrack);
  } else {
    return estimateNFittedTrack(cachingVertex, vertexTrack);
  }
}



// methods to calculate track<-->vertex compatibility
// with the track belonging to the vertex

template <unsigned int N>
typename KalmanVertexTrackCompatibilityEstimator<N>::BDpair
KalmanVertexTrackCompatibilityEstimator<N>::estimateFittedTrack
		(const CachingVertex<N> & v, const RefCountedVertexTrack track) const
{
  //remove track from the vertex using the vertex updator
  // Using the update instead of the remove methode, we can specify a weight which
  // is different than then one which the vertex track has been defined with.
  //CachingVertex rVert = updator.remove(v, track);
  RefCountedVertexTrack newSmoothedTrack = trackUpdator.update(v, track);
//   std::cout << newSmoothedTrack->smoothedChi2()<<" "<<estimateDifference(v,rVert,newSmoothedTrack)<<std::endl;
//   return estimateDifference(v,rVert,newSmoothedTrack);
  return BDpair(true, newSmoothedTrack->smoothedChi2());
}

// method calculating track<-->vertex compatibility
//with the track not belonging to vertex
template <unsigned int N>
typename KalmanVertexTrackCompatibilityEstimator<N>::BDpair
KalmanVertexTrackCompatibilityEstimator<N>::estimateNFittedTrack
 	(const CachingVertex<N> & v, const RefCountedVertexTrack track) const
{
  // Using the update instead of the add methode, we can specify a weight which
  // is different than then one which the vertex track has been defined with.
  CachingVertex<N> rVert = updator.add(v, track);
  if (!rVert.isValid()) return BDpair(false,-1.);
  return BDpair(true, rVert.totalChiSquared()-v.totalChiSquared());
}   



template <unsigned int N>
typename KalmanVertexTrackCompatibilityEstimator<N>::BDpair
KalmanVertexTrackCompatibilityEstimator<N>::estimateDifference
	(const CachingVertex<N> & more, const CachingVertex<N> & less, 
         const RefCountedVertexTrack track) const
{
  BDpair trackRes = helper.trackParameterChi2(track);
  return BDpair(trackRes.first, trackRes.second + helper.vertexChi2(less, more)) ;
}

template class KalmanVertexTrackCompatibilityEstimator<5>;
