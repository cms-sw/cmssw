#include "RecoVertex/VertexTools/interface/SequentialVertexSmoother.h"


SequentialVertexSmoother::SequentialVertexSmoother(
  const VertexTrackUpdator & vtu, 
  const VertexSmoothedChiSquaredEstimator & vse, 
  const TrackToTrackCovCalculator & covCalc) :
  theVertexTrackUpdator(vtu.clone()), 
  theVertexSmoothedChiSquaredEstimator(vse.clone()), 
  theTrackToTrackCovCalculator(covCalc.clone()) 
{}


SequentialVertexSmoother::~SequentialVertexSmoother() 
{
  delete theVertexTrackUpdator;
  delete theVertexSmoothedChiSquaredEstimator;
  delete theTrackToTrackCovCalculator;
}


SequentialVertexSmoother::SequentialVertexSmoother(
  const SequentialVertexSmoother & smoother) 
{
  theVertexTrackUpdator = smoother.vertexTrackUpdator()->clone();
  theVertexSmoothedChiSquaredEstimator 
    = smoother.vertexSmoothedChiSquaredEstimator()->clone();
  theTrackToTrackCovCalculator = smoother.trackToTrackCovCalculator()->clone();
}


CachingVertex
SequentialVertexSmoother::smooth(const CachingVertex & vertex) const
{

  // Track refit

  vector<RefCountedVertexTrack> newTracks;
  if (theVertexTrackUpdator != 0) {
    const vector<RefCountedVertexTrack>&  vOut=vertex.tracks();
    for(vector<RefCountedVertexTrack>::const_iterator i = vOut.begin();
  	  i != vOut.end();i++)
    {
      RefCountedVertexTrack nTrack = theVertexTrackUpdator->update(vertex, *i);
      newTracks.push_back(nTrack);
    }
  } else {
    newTracks = vertex.tracks();
  }

  // intermediate vertex for chi2 calculation and TktoTkcovariance map
  CachingVertex interVertex(vertex.position(), vertex.weight(),
  				newTracks, 0.);

  // Smoothed chi2

  float smChi2 = vertex.totalChiSquared();
  if (theVertexSmoothedChiSquaredEstimator != 0) {
    smChi2 = theVertexSmoothedChiSquaredEstimator->estimate(interVertex);
  }

  if (theTrackToTrackCovCalculator == 0) {
    if  (vertex.hasPrior()) {
      return CachingVertex(vertex.priorVertexState(), vertex.vertexState(),
    		  newTracks, smChi2);
    } else {
      return CachingVertex(vertex.vertexState(), newTracks, smChi2);
    }
  }
  
  //TktoTkcovariance map
  TrackToTrackMap tkMap = (*theTrackToTrackCovCalculator)(interVertex);

//   CachingVertex finalVertex(vertex.position(), vertex.error(),
// 			    newTracks, smChi2, tkMap);
  if  (vertex.hasPrior()) {
    CachingVertex finalVertex(vertex.priorVertexState(), vertex.vertexState(),
    		newTracks, smChi2, tkMap);
    return finalVertex;
  }

  CachingVertex finalVertex(vertex.vertexState(), newTracks, smChi2, tkMap);
  return finalVertex;

}
