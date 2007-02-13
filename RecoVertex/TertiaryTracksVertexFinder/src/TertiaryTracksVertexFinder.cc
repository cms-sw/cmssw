
#include "RecoVertex/TertiaryTracksVertexFinder/interface/TertiaryTracksVertexFinder.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"


TertiaryTracksVertexFinder::TertiaryTracksVertexFinder() 
{
  KalmanVertexFitter vf(false);
  KalmanVertexUpdator vu;
  KalmanVertexTrackCompatibilityEstimator ve;

  theFinder = new ConfigurableTertiaryTracksVertexFinder (&vf, &vu, &ve);
}


TertiaryTracksVertexFinder::~TertiaryTracksVertexFinder() 
{
  delete theFinder;
}


