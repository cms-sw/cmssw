#include "RecoVertex/VertexTools/interface/SequentialVertexSmoother.h"


template <unsigned int N>
SequentialVertexSmoother<N>::SequentialVertexSmoother(
  const VertexTrackUpdator<N> & vtu, 
  const VertexSmoothedChiSquaredEstimator<N> & vse, 
  const TrackToTrackCovCalculator<N> & covCalc) :
  theVertexTrackUpdator(vtu.clone()), 
  theVertexSmoothedChiSquaredEstimator(vse.clone()), 
  theTrackToTrackCovCalculator(covCalc.clone()) 
{}


template <unsigned int N>
SequentialVertexSmoother<N>::~SequentialVertexSmoother() 
{
  delete theVertexTrackUpdator;
  delete theVertexSmoothedChiSquaredEstimator;
  delete theTrackToTrackCovCalculator;
}


template <unsigned int N>
SequentialVertexSmoother<N>::SequentialVertexSmoother(
  const SequentialVertexSmoother & smoother) 
{
  theVertexTrackUpdator = smoother.vertexTrackUpdator()->clone();
  theVertexSmoothedChiSquaredEstimator 
    = smoother.vertexSmoothedChiSquaredEstimator()->clone();
  theTrackToTrackCovCalculator = smoother.trackToTrackCovCalculator()->clone();
}


template <unsigned int N>
CachingVertex<N>
SequentialVertexSmoother<N>::smooth(const CachingVertex<N> & vertex) const
{

  // Track refit

  std::vector<RefCountedVertexTrack> newTracks;
  if (theVertexTrackUpdator != nullptr) {
    const std::vector<RefCountedVertexTrack>&  vOut=vertex.tracks();
    for(typename std::vector<RefCountedVertexTrack>::const_iterator i = vOut.begin();
  	  i != vOut.end();i++)
    {
      RefCountedVertexTrack nTrack = theVertexTrackUpdator->update(vertex, *i);
      newTracks.push_back(nTrack);
    }
  } else {
    newTracks = vertex.tracks();
  }

  // intermediate vertex for chi2 calculation and TktoTkcovariance map
  CachingVertex<N> interVertex(vertex.position(), vertex.weight(),
  				newTracks, 0.);
  if ( vertex.hasPrior() )
  {
    interVertex = CachingVertex<N> ( vertex.priorPosition(), vertex.priorError(), 
        vertex.position(), vertex.weight(), newTracks, 0.); 
  }

  // Smoothed chi2

  float smChi2 = vertex.totalChiSquared();
  if (theVertexSmoothedChiSquaredEstimator != nullptr) {
    std::pair<bool, double> result = theVertexSmoothedChiSquaredEstimator->estimate(interVertex);
    smChi2 = result.second;
  }

  if (theTrackToTrackCovCalculator == nullptr) {
    if  (vertex.hasPrior()) {
      return CachingVertex<N>(vertex.priorVertexState(), vertex.vertexState(),
    		  newTracks, smChi2);
    } else {
      return CachingVertex<N>(vertex.vertexState(), newTracks, smChi2);
    }
  }
  
  //TktoTkcovariance map
  typename CachingVertex<N>::TrackToTrackMap tkMap = (*theTrackToTrackCovCalculator)(interVertex);

//   CachingVertex<N> finalVertex(vertex.position(), vertex.error(),
// 			    newTracks, smChi2, tkMap);
  if  (vertex.hasPrior()) {
    CachingVertex<N> finalVertex(vertex.priorVertexState(), vertex.vertexState(),
    		newTracks, smChi2, tkMap);
    return finalVertex;
  }

  CachingVertex<N> finalVertex(vertex.vertexState(), newTracks, smChi2, tkMap);
  return finalVertex;

}

template class SequentialVertexSmoother<5>;
template class SequentialVertexSmoother<6>;
