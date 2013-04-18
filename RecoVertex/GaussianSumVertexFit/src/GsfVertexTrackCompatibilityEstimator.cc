#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexTrackCompatibilityEstimator.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include <algorithm>
using namespace reco;

// template <unsigned int N>
// struct vT_find
// {
//   typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
//   bool operator()(const CachingVertex<5> & v, const RefCountedVertexTrack t)
//   {
// //initial tracks  
//     vector<RefCountedVertexTrack> tracks = v.tracks();
//     vector<RefCountedVertexTrack>::iterator pos 
//       = find_if(tracks.begin(), tracks.end(), VertexTrackEqual(t));
//     return (pos != tracks.end());
//   }
// }; 
 
 
std::pair<bool, double> 
GsfVertexTrackCompatibilityEstimator::estimate(const CachingVertex<5> & vertex,
					       const RefCountedVertexTrack tr, unsigned int hint) const
{
//checking if the track passed really belongs to the vertex
  std::vector<RefCountedVertexTrack> tracks = vertex.tracks();
  std::vector<RefCountedVertexTrack>::iterator pos 
    = find_if(tracks.begin(), tracks.end(), VertexTrackEqual<5>(tr));
 if (pos != tracks.end()) {
   return estimateFittedTrack(vertex,*pos);
 } else {
   return estimateNFittedTrack(vertex,tr);
 }
} 


std::pair<bool, double>
GsfVertexTrackCompatibilityEstimator::estimate(const CachingVertex<5> & vertex, 
					       const RefCountedLinearizedTrackState track,
					       unsigned int hint) const
{
  RefCountedVertexTrack vertexTrack = vTrackFactory.vertexTrack(track,
 						 vertex.vertexState());
  return estimate(vertex, vertexTrack);
}


std::pair<bool, double>
GsfVertexTrackCompatibilityEstimator::estimate(const reco::Vertex & vertex, 
			 const reco::TransientTrack & track) const
{ 	
//   GlobalPoint linP(vertex.position().x(), vertex.position().z(),vertex.position().z());
    GlobalPoint linP(Basic3DVector<float> (vertex.position()));

  RefCountedLinearizedTrackState linTrack = 
  			lTrackFactory.linearizedTrackState(linP, track);
  GlobalError err(vertex.covariance());
  VertexState vState(linP, err);
  RefCountedVertexTrack vertexTrack = vTrackFactory.vertexTrack(linTrack, vState);

  std::vector<RefCountedVertexTrack> initialTracks(1, vertexTrack);
  CachingVertex<5> cachingVertex(linP, err, initialTracks,
  			    vertex.chi2());
  // FIXME: this should work also for tracks without a persistent ref.
//   const TrackTransientTrack* ttt = dynamic_cast<const TrackTransientTrack*>(track.basicTransientTrack());
//   if ((ttt!=0) && 
  if (find(vertex.tracks_begin(), vertex.tracks_end(), track.trackBaseRef()) != vertex.tracks_end())
  {
    return estimateFittedTrack(cachingVertex, vertexTrack);
  } else {
    return estimateNFittedTrack(cachingVertex, vertexTrack);
  }
}



// methods to calculate track<-->vertex compatibility
// with the track belonging to the vertex

std::pair<bool, double> 
GsfVertexTrackCompatibilityEstimator::estimateFittedTrack
		(const CachingVertex<5> & v, const RefCountedVertexTrack track) const
{
  //remove track from the vertex using the vertex updator
  // Using the update instead of the remove methode, we can specify a weight which
  // is different than then one which the vertex track has been defined with.
  if (track->refittedStateAvailable()) return BDpair(true, track->smoothedChi2());
  throw VertexException
    ("GsfVertexTrackCompatibilityEstimator::vertex has to be smoothed.");

//   CachingVertex rVert = updator.remove(v, track);
//   RefCountedVertexTrack newSmoothedTrack = trackUpdator.update(v, track);
//   return estimateDifference(v,rVert,newSmoothedTrack);
}

// method calculating track<-->vertex compatibility
//with the track not belonging to vertex
std::pair<bool, double>
GsfVertexTrackCompatibilityEstimator::estimateNFittedTrack
 	(const CachingVertex<5> & v, const RefCountedVertexTrack track) const
{
  // Using the update instead of the add methode, we can specify a weight which
  // is different than then one which the vertex track has been defined with.
  CachingVertex<5> rVert = updator.add(v, track);
  if (!rVert.isValid()) return BDpair(false,-1.);
  return BDpair(true, rVert.totalChiSquared()-v.totalChiSquared());
}   

