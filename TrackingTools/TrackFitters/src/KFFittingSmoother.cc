#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
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
    //else {
    //LogTrace("TrackFitters") << "dump hits after smoothing";
    //Trajectory::DataContainer meas = smoothed[0].measurements();
    //for (Trajectory::DataContainer::iterator it=meas.begin();it!=meas.end();++it) {
    //LogTrace("TrackFitters") << "hit #" << meas.end()-it-1 << " validity=" << it->recHit()->isValid() 
    //<< " det=" << it->recHit()->geographicalId().rawId();
    //}
    //}

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

	//replace outlier hit with invalid hit
	for (unsigned int j=0;j<myHits.size();++j) { 
	  if (outlierId == myHits[j]->geographicalId().rawId()){
	    LogTrace("TrackFitters") << "Rejecting outlier hit  with estimate " << cut << " at position " 
					 << j << " with rawId=" << myHits[j]->geographicalId().rawId();
	    LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
	    myHits[j] = InvalidTransientRecHit::build(outlierDet, TrackingRecHit::missing);
	  }
	}

	//look if there are two consecutive invalid hits
	if (breakTrajWith2ConsecutiveMissing) {
	  unsigned int firstinvalid = myHits.size()-1;
	  for (unsigned int j=0;j<myHits.size()-1;++j) { 
	    if (myHits[j]->type()==TrackingRecHit::missing && myHits[j+1]->type()==TrackingRecHit::missing) {
	      firstinvalid = j;
	      LogTrace("TrackFitters") << "Found two consecutive missing hits. First invalid: " << firstinvalid;
	      break;
	    }
	  }
	  //reject all the hits after the last valid before two consecutive invalid (missing) hits
	  //hits are sorted in the same order as in the track candidate FIXME??????
	  myHits.erase(myHits.begin()+firstinvalid,myHits.end());
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

    if (noInvalidHitsBeginEnd) {

      // discard latest dummy measurements
      if (!smoothed[0].empty() && !smoothed[0].lastMeasurement().recHit()->isValid()) 
	LogTrace("TrackFitters") << "Last measurement is invalid";
      while (!smoothed[0].empty() && !smoothed[0].lastMeasurement().recHit()->isValid()) smoothed[0].pop();
      
      //remove the invalid hits at the begin of the trajectory
      if(!smoothed[0].firstMeasurement().recHit()->isValid()) {
	LogTrace("TrackFitters") << "First measurement is invalid";
	Trajectory tmpTraj(smoothed[0].seed(),smoothed[0].direction());
	Trajectory::DataContainer meas = smoothed[0].measurements();
	
	for (Trajectory::DataContainer::iterator it=meas.begin();it!=meas.end();++it) {
	  if (!it->recHit()->isValid()) continue;
	  else {
	    LogTrace("TrackFitters") << "First valid measurement is: " << it-meas.begin();
	    const KFTrajectorySmoother* myKFSmoother = dynamic_cast<const KFTrajectorySmoother*>(smoother());
	    if (!myKFSmoother) throw cms::Exception("TrackFitters") << "trying to use outliers rejection with a smoother different from KFTrajectorySmoother. Please disable outlier rejection.";
	    const MeasurementEstimator* estimator = myKFSmoother->estimator();
	    for (Trajectory::DataContainer::iterator itt=it;itt!=meas.end();++itt) {
	      if (itt->recHit()->isValid()) 
		tmpTraj.push(*itt,estimator->estimate(itt->backwardPredictedState(),*itt->recHit()).second);//chi2!!!!!!
	      else tmpTraj.push(*itt);
	    }
	    break;
	  }
	}
	smoothed.clear();
	smoothed.push_back(tmpTraj);
	
      }
    }
    
    LogTrace("TrackFitters") << "end: returning smoothed trajectory with chi2=" 
				 << smoothed[0].chiSquared() ;

    //LogTrace("TrackFitters") << "dump hits before return";
    //Trajectory::DataContainer meas = smoothed[0].measurements();
    //for (Trajectory::DataContainer::iterator it=meas.begin();it!=meas.end();++it) {
    //LogTrace("TrackFitters") << "hit #" << meas.end()-it-1 << " validity=" << it->recHit()->isValid() 
    //<< " det=" << it->recHit()->geographicalId().rawId();
    //}

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
