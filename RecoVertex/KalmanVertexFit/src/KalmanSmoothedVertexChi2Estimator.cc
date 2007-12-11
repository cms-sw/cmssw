#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
// #include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"


template <unsigned int N>
float KalmanSmoothedVertexChi2Estimator<N>::estimate(const CachingVertex<N> & vertex) const
{
//initial vertex part
  float v_part = 0.;
  float returnChi = 0.;
  
  if (vertex.hasPrior()) {
    v_part = helper.vertexChi2(vertex.priorVertexState(), vertex.vertexState());
  }
 
//vector of tracks part
  typedef typename CachingVertex<N>::RefCountedVertexTrack RefCountedVertexTrack;
  vector< RefCountedVertexTrack > tracks = vertex.tracks();
  float sum = 0.;
  for(typename vector<RefCountedVertexTrack>::iterator i = tracks.begin(); i != tracks.end(); i++)
  {
   sum += (*i)->weight() * helper.trackParameterChi2((*i)->linearizedTrack(), (*i)->refittedState());
  }
 returnChi = v_part + sum;
 return returnChi;   
}

template class KalmanSmoothedVertexChi2Estimator<5>;
template class KalmanSmoothedVertexChi2Estimator<6>;
