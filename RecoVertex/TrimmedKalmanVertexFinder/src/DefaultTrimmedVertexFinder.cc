#include "Utilities/Configuration/interface/Architecture.h"

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/DefaultTrimmedVertexFinder.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"


DefaultTrimmedVertexFinder::DefaultTrimmedVertexFinder() 
{
  KalmanVertexFitter vf(false);
  KalmanVertexUpdator vu;
  KalmanVertexTrackCompatibilityEstimator ve;

  theFinder = new ConfigurableTrimmedVertexFinder (&vf, &vu, &ve);
}


DefaultTrimmedVertexFinder::~DefaultTrimmedVertexFinder() 
{
  delete theFinder;
}


DefaultTrimmedVertexFinder::DefaultTrimmedVertexFinder(
  const DefaultTrimmedVertexFinder & other)
{
  theFinder = other.theFinder->clone();
}
