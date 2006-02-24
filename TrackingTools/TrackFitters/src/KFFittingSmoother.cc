#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"

KFFittingSmoother::KFFittingSmoother()
{
  //
  // Temporary: take mass from same configurable as the MaterialEffectsUpdator
  //
  //static SimpleConfigurable<float> massConfigurable(0.1057,"MaterialEffects:defaultMass");
  //init(massConfigurable.value());
  init(0.1057);//temporary hack
}

KFFittingSmoother::KFFittingSmoother(const float mass)
{
  init(mass);
}

void KFFittingSmoother::init (const float mass)
{
  PropagatorWithMaterial propagator(alongMomentum,mass);
  KFUpdator updator;
  Chi2MeasurementEstimator estimator(100);
  theFitter = new KFTrajectoryFitter(propagator,updator,estimator);

  propagator.setPropagationDirection(oppositeToMomentum);
  theSmoother = new KFTrajectorySmoother(propagator,updator,estimator);
}

KFFittingSmoother::~KFFittingSmoother() {

  delete theSmoother;
  delete theFitter;

}


vector<Trajectory> 
KFFittingSmoother::trajectories(const Trajectory& t) const {

  if(!t.isValid()) return vector<Trajectory>();

  vector<Trajectory> fitted = fitter()->fit(t);
  return smoothingStep(fitted);
}

vector<Trajectory> KFFittingSmoother::
trajectories(const TrajectorySeed& aSeed,
	     const edm::OwnVector<TransientTrackingRecHit>& hits, 
	     const TrajectoryStateOnSurface& firstPredTsos) const 
{
  if(hits.empty()) return vector<Trajectory>();
  
  vector<Trajectory> fitted = 
    fitter()->fit(aSeed, hits, tsosWithError(firstPredTsos));
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

  return result;
}
