#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"


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
	     const edm::OwnVector<TransientTrackingRecHit>& hits, 
	     const TrajectoryStateOnSurface& firstPredTsos) const 
{
  if(hits.empty()) return vector<Trajectory>();
  
  vector<Trajectory> fitted = 
    fitter()->fit(aSeed, hits, tsosWithError(firstPredTsos));
  //    std::cout <<" ))))))))))))))))) IN KFFittingSmoother::fit "<<fitted.size()<<std::endl;
    
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
  //  std::cout <<" ))))))))))))))))) IN KFFittingSmoother::smoothingStep "<<result.size()<<std::endl;
  return result;
}
vector<Trajectory> KFFittingSmoother::fit(const TrajectorySeed& aSeed,
					   const edm::OwnVector<TransientTrackingRecHit>& hits) const{

  throw cms::Exception("TrackingTools/TrackFitters", 
		       "KFFittingSmoother::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return vector<Trajectory>();
}
