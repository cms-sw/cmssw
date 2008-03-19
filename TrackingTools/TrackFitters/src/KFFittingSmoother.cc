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
  vector<Trajectory> smoothed;
  if(t.isValid()) { 
    vector<Trajectory> fitted = fitter()->fit(t);
    smoothingStep(fitted, smoothed);
  }
  return smoothed;
}

vector<Trajectory> KFFittingSmoother::
fit(const TrajectorySeed& aSeed,
    const RecHitContainer& hits, 
    const TrajectoryStateOnSurface& firstPredTsos) const 
{
  LogDebug("TrackFitters") << "In KFFittingSmoother::fit";

  //if(hits.empty()) return vector<Trajectory>(); // gio: moved later to optimize return value
  
  bool hasoutliers;
  RecHitContainer myHits = hits; 
  vector<Trajectory> smoothed;
  vector<Trajectory> tmp_first;

  do{
    if(hits.empty()) { smoothed.clear(); break; }

    //if no outliers the fit is done only once
    //for (unsigned int j=0;j<myHits.size();j++) { 
    //if (myHits[j]->det()) 
    //LogTrace("TrackFitters") << "hit #:" << j+1 << " rawId=" << myHits[j]->det()->geographicalId().rawId() 
    //<< " validity=" << myHits[j]->isValid();
    //else
    //LogTrace("TrackFitters") << "hit #:" << j+1 << " Hit with no Det information";
    //}

    hasoutliers = false;
    double cut = theEstimateCut;
    unsigned int outlierId = 0;
    const GeomDet* outlierDet = 0;

    //call the fitter
    vector<Trajectory> fitted = fitter()->fit(aSeed, myHits, firstPredTsos);
    //call the smoother
    smoothed.clear();
    smoothingStep(fitted, smoothed);

    //if (tmp_first.size()==0) tmp_first = smoothed; moved later
    
    if (smoothed.empty()) {
	if (rejectTracksFlag){
	  LogTrace("TrackFitters") << "smoothed.size()==0 => trajectory rejected";
	  //return vector<Trajectory>(); // break is enough to get this
	} else {
	  LogTrace("TrackFitters") << "smoothed.size()==0 => returning orignal trajectory" ;
          smoothed.swap(tmp_first); // if first attempt, tmp_first would be empty anyway
	}
        break;
    }

    if (theEstimateCut>0) {
      if (smoothed[0].foundHits()<theMinNumberOfHits) {
	if (rejectTracksFlag){
	  LogTrace("TrackFitters") << "smoothed[0].foundHits()<theMinNumberOfHits => trajectory rejected";
          smoothed.clear();
	  //return vector<Trajectory>(); // break is enough
	} else {
          // it might be it's the first step
          if (!tmp_first.empty()) { tmp_first.swap(smoothed); } 
	  LogTrace("TrackFitters") 
	    << "smoothed[0].foundHits()<theMinNumberOfHits => returning orignal trajectory with chi2=" 
	    <<  smoothed[0].chiSquared() ;
	}
        break;
      }
      //check if there are outliers
      const std::vector<TrajectoryMeasurement> & vtm = smoothed[0].measurements();
      for (std::vector<TrajectoryMeasurement>::const_iterator tm=vtm.begin(); tm!=vtm.end();tm++){
	double estimate = tm->estimate();
	if (estimate > cut) {
	  hasoutliers = true;
	  cut = estimate;
	  outlierId  = tm->recHit()->geographicalId().rawId();
          outlierDet = tm->recHit()->det();
	}
      }
      if (hasoutliers) {//reject outliers
	for (unsigned int j=0;j<myHits.size();++j) 
	  if (outlierId == myHits[j]->geographicalId().rawId()){
	    LogTrace("TrackFitters") << "Rejecting outlier hit  with estimate " << cut << " at position " 
				     << j << " with rawId=" << myHits[j]->geographicalId().rawId();
	    LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
	    myHits[j] = InvalidTransientRecHit::build(outlierDet, TrackingRecHit::missing);
	  }
      }
    }
    if ( hasoutliers &&        // otherwise there won't be a 'next' loop where tmp_first could be useful 
         !rejectTracksFlag &&  // othewrise we won't ever need tmp_first
         tmp_first.empty() ) { // only at first step
        smoothed.swap(tmp_first);
    }
         
  } while(hasoutliers);
  if (!smoothed.empty()) {
      LogTrace("TrackFitters") << "end: returning smoothed trajectory with chi2=" 
                               << smoothed[0].chiSquared() ;
  }
  return smoothed;
}


void 
KFFittingSmoother::smoothingStep(vector<Trajectory>& fitted, vector<Trajectory> &smoothed) const
{
 
  for(vector<Trajectory>::iterator it = fitted.begin(); it != fitted.end();
      it++) {
    vector<Trajectory> mysmoothed = smoother()->trajectories(*it);
    smoothed.insert(smoothed.end(), mysmoothed.begin(), mysmoothed.end());
  }
  LogDebug("TrackFitters") << "In KFFittingSmoother::smoothingStep "<<smoothed.size();
}

vector<Trajectory> KFFittingSmoother::fit(const TrajectorySeed& aSeed,
					   const RecHitContainer& hits) const{

  throw cms::Exception("TrackFitters", 
		       "KFFittingSmoother::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return vector<Trajectory>();
}
