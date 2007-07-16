#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

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
  LogDebug("TrackFitters") << "In KFFittingSmoother::fit";
  if(hits.empty()) return vector<Trajectory>();
  
  bool hasoutliers;
  RecHitContainer myHits = hits; 
  vector<Trajectory> smoothed;

  do{//if no outliers the fit is done only once
    for (unsigned int j=0;j<myHits.size();j++) { 
      if (myHits[j]->det()) 
	LogTrace("TrackFitters") << "hit #:" << j << " rawId=" << myHits[j]->det()->geographicalId().rawId() 
				 << " validity=" << myHits[j]->isValid();
      else
	LogTrace("TrackFitters") << "hit #:" << j << " Hit with no Det information";
    }
    hasoutliers = false;
    double cut = theEstimateCut;
    unsigned int outlierId = 0;
    const GeomDet* outlierDet = 0;

    //call the fitter
    vector<Trajectory> fitted = fitter()->fit(aSeed, myHits, firstPredTsos);
    //call the smoother
    smoothed = smoothingStep(fitted);

    if (smoothed.size()==0) return vector<Trajectory>();
    if (theEstimateCut>0){
    if (smoothed[0].foundHits()<theMinNumberOfHits) return vector<Trajectory>();
      //check if there are outliers
      std::vector<TrajectoryMeasurement> vtm = smoothed[0].measurements();
      for (std::vector<TrajectoryMeasurement>::iterator tm=vtm.begin(); tm!=vtm.end();tm++){
	double estimate = tm->estimate();
	if (estimate > cut) {
	  hasoutliers = true;
	  cut = estimate;
	  outlierId = tm->recHit()->det()->geographicalId().rawId();
	  outlierDet = tm->recHit()->det();
	}
      }
      if (hasoutliers) {//reject outliers
	for (unsigned int j=0;j<myHits.size();++j) 
	  if (myHits[j]->det()) if (outlierId==myHits[j]->det()->geographicalId().rawId()){
	    LogDebug("TrackFitters") << "Rejecting outlier hit  with estimate " << cut << " at position " 
				     << j << " with rawId=" << myHits[j]->det()->geographicalId().rawId();
	    LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
	    myHits.erase(myHits.begin()+j);
	    myHits.insert(myHits.begin()+j,InvalidTransientRecHit::build(outlierDet, TrackingRecHit::missing) );
	  }
      }
    }
  } while(hasoutliers);
  return smoothed;
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
  LogDebug("TrackFitters") << "In KFFittingSmoother::smoothingStep "<<result.size();
  return result;
}
vector<Trajectory> KFFittingSmoother::fit(const TrajectorySeed& aSeed,
					   const RecHitContainer& hits) const{

  throw cms::Exception("TrackFitters", 
		       "KFFittingSmoother::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return vector<Trajectory>();
}
