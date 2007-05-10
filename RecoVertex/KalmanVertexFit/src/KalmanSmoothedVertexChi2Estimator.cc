#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"


float KalmanSmoothedVertexChi2Estimator::estimate(const CachingVertex & vertex) const
{
//initial vertex part
  float v_part = 0.;
  float returnChi = 0.;
  
  if (vertex.hasPrior()) {
    v_part = helper.vertexChi2(vertex.priorVertexState(), vertex.vertexState());
  }
 
//vector of tracks part
  vector<RefCountedVertexTrack> tracks = vertex.tracks();
  float sum = 0.;
  for(vector<RefCountedVertexTrack>::iterator i = tracks.begin(); i != tracks.end(); i++)
  {
   sum += helper.trackParameterChi2((*i)->linearizedTrack(), (*i)->refittedState());
  }
  returnChi = v_part + sum;
  return returnChi;
}
