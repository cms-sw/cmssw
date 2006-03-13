#include "Utilities/Configuration/interface/Architecture.h"

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"


KalmanTrimmedVertexFinder::KalmanTrimmedVertexFinder() 
{
  KalmanVertexFitter vf(false);
  KalmanVertexUpdator vu;
  KalmanVertexTrackCompatibilityEstimator ve;

  theFinder = new ConfigurableTrimmedVertexFinder (&vf, &vu, &ve);
}


KalmanTrimmedVertexFinder::~KalmanTrimmedVertexFinder() 
{
  delete theFinder;
}


KalmanTrimmedVertexFinder::KalmanTrimmedVertexFinder(
  const KalmanTrimmedVertexFinder & other)
{
  theFinder = other.theFinder->clone();
}
