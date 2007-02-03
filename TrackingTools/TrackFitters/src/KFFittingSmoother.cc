#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

KFFittingSmoother::~KFFittingSmoother() {
  delete theSmoother;
  delete theFitter;
}


vector<Trajectory> 
KFFittingSmoother::fit(const Trajectory& t) const {
  
  if(!t.isValid()) return vector<Trajectory>();

  vector<Trajectory> fitted = fitter()->fit(t);
  return smoothingStep(fitted);
}

vector<Trajectory> KFFittingSmoother::
fit(const TrajectorySeed& aSeed,
	     const RecHitContainer& hits, 
	     const TrajectoryStateOnSurface& firstPredTsos) const 
{
  if(hits.empty()) return vector<Trajectory>();
  
  vector<Trajectory> fitted = 
    //    fitter()->fit(aSeed, hits, tsosWithError(firstPredTsos));
    fitter()->fit(aSeed, hits, firstPredTsos);
  LogDebug("TrackingTools/TrackFitters") << "In KFFittingSmoother::fit "<<fitted.size();
   
   return smoothingStep(fitted);
}


vector<Trajectory> 
KFFittingSmoother::smoothingStep(vector<Trajectory>& fitted) const
{
  vector<Trajectory> result; 
  
  for(vector<Trajectory>::iterator it = fitted.begin(); it != fitted.end();
      it++) {
    vector<Trajectory> smoothed = smoother()->trajectories(*it);
    result.insert(result.end(), smoothed.begin(), smoothed.end());
  }
  LogDebug("TrackingTools/TrackFitters") << "In KFFittingSmoother::smoothingStep "<<result.size();
  return result;
}
vector<Trajectory> KFFittingSmoother::fit(const TrajectorySeed& aSeed,
					   const RecHitContainer& hits) const{

  throw cms::Exception("TrackingTools/TrackFitters", 
		       "KFFittingSmoother::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return vector<Trajectory>();
}
