#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
// #include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

// ggiurgiu@fnal.gov: Add headers needed to cut on pixel hit probability
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

using namespace std;

Trajectory KFFittingSmoother::fitOne(const Trajectory& t, fitType type) const {
  if (!t.isValid() ) return Trajectory();
  return smoothingStep(theFitter->fitOne(t,type));
}

bool KFFittingSmoother::checkForNans(const Trajectory & theTraj) const
{
  if (edm::isNotFinite(theTraj.chiSquared ())) return false;
  const std::vector<TrajectoryMeasurement> & vtm = theTraj.measurements();
  for (std::vector<TrajectoryMeasurement>::const_iterator tm=vtm.begin(); tm!=vtm.end();tm++) {
    if (edm::isNotFinite(tm->estimate())) return false;
    AlgebraicVector5 v = tm->updatedState ().localParameters ().vector();
    for (int i=0;i<5;++i) if (edm::isNotFinite(v[i])) return false;
    const AlgebraicSymMatrix55 & m = tm->updatedState ().curvilinearError ().matrix();
    for (int i=0;i<5;++i)
      for (int j=0;j<(i+1);++j) if (edm::isNotFinite(m(i,j))) return false;
  }
  return true;
}

Trajectory KFFittingSmoother::fitOne(const TrajectorySeed& aSeed,
				  const RecHitContainer& hits,
				  const TrajectoryStateOnSurface& firstPredTsos,
				  fitType type) const
{
  LogDebug("TrackFitters") << "In KFFittingSmoother::fit";

  if ( hits.empty() ) return Trajectory();

  bool hasoutliers=false;
  bool has_low_pixel_prob=false; // ggiurgiu@fnal.gov: Add flag for pixel hits with low template probability

  // ggiurgiu@fnal.gov: If log(Prob) < -15.0 or if Prob <= 0.0 then set log(Prob) = -15.0
  double log_pixel_probability_lower_limit = -15.0;

  RecHitContainer myHits = hits;
  Trajectory smoothed;
  Trajectory tmp_first;
  bool firstTry=true;

  do {

    //if no outliers the fit is done only once
    //for (unsigned int j=0;j<myHits.size();j++) {
    //if (myHits[j]->det())
    //LogTrace("TrackFitters") << "hit #:" << j+1 << " rawId=" << myHits[j]->det()->geographicalId().rawId()
    //<< " validity=" << myHits[j]->isValid();
    //else
    //LogTrace("TrackFitters") << "hit #:" << j+1 << " Hit with no Det information";
    //}

    hasoutliers        = false;
    has_low_pixel_prob = false; // ggiurgiu@fnal.gov

    double cut = theEstimateCut;

    double log_pixel_prob_cut = theLogPixelProbabilityCut;  // ggiurgiu@fnal.gov


    unsigned int outlierId = 0;
    const GeomDet* outlierDet = 0;

    unsigned int low_pixel_prob_Id = 0; // ggiurgiu@fnal.gov
    const GeomDet* low_pixel_prob_Det = 0; // ggiurgiu@fnal.gov

    //call the fitter
    smoothed  = smoothingStep(theFitter->fitOne(aSeed, myHits, firstPredTsos));

    //if (tmp_first.size()==0) tmp_first = smoothed; moved later

    if ( !smoothed.isValid() )  {
      if ( rejectTracksFlag ) {
	LogTrace("TrackFitters") << "smoothed invalid => trajectory rejected";
	//return vector<Trajectory>(); // break is enough to get this
      } else {
	LogTrace("TrackFitters") << "smoothed invalid => returning orignal trajectory" ;
	std::swap(smoothed, tmp_first); // if first attempt, tmp_first would be invalid anyway
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

    if (!checkForNans(smoothed)) {
      edm::LogError("TrackNaN")<<"Track has NaN";
      return Trajectory();
    }


    if ( theEstimateCut > 0 || log_pixel_prob_cut > log_pixel_probability_lower_limit ) {
      if ( smoothed.foundHits() < theMinNumberOfHits ) {
	if ( rejectTracksFlag ) {
	  LogTrace("TrackFitters") << "smoothed.foundHits()<theMinNumberOfHits => trajectory rejected";
	  return Trajectory(); // invalid
	} else{
	  // it might be it's the first step
	  if ( firstTry ) {
	    firstTry = false;
	    std::swap(tmp_first,smoothed);
	  }

	  LogTrace("TrackFitters")
	    << "smoothed.foundHits()<theMinNumberOfHits => returning orignal trajectory with chi2="
	    <<  smoothed.chiSquared() ;
	}
	break;
      }

      // Check if there are outliers or low probability pixel rec hits
      const std::vector<TrajectoryMeasurement> & vtm = smoothed.measurements();

      double log_pixel_hit_probability = -999999.9;

      for (std::vector<TrajectoryMeasurement>::const_iterator tm=vtm.begin(); tm!=vtm.end();tm++) {
	double estimate = tm->estimate();

	// --- here is the block of code about generic chi2-based Outlier Rejection ---
	if ( estimate > cut && theEstimateCut > 0 ) {
	  hasoutliers = true;
	  cut = estimate;
	  outlierId  = tm->recHit()->geographicalId().rawId();
	  outlierDet = tm->recHit()->det();
	}
	// --- here the block of code about generic chi2-based Outlier Rejection ends ---


	// --- here is the block of code about PXL Outlier Rejection ---
	if(log_pixel_prob_cut > log_pixel_probability_lower_limit){
	  // TO BE FIXED: the following code should really be moved into an external class or
	  // at least in a separate function. Current solution is ugly!
	  // The KFFittingSmoother shouldn't handle the details of
	  // Outliers identification and rejection. It shoudl just fit tracks.

	  // ggiurgiu@fnal.gov: Get pixel hit probability here
	  TransientTrackingRecHit::ConstRecHitPointer hit = tm->recHit();
	  unsigned int testSubDetID = ( hit->geographicalId().subdetId() );

	  if ( hit->isValid() &&
	       hit->geographicalId().det() == DetId::Tracker &&
	       ( testSubDetID == PixelSubdetector::PixelBarrel ||
		 testSubDetID == PixelSubdetector::PixelEndcap )
	       ){
	    // get the enclosed persistent hit
	    const TrackingRecHit* persistentHit = hit->hit();

	    // check if it's not null, and if it's a valid pixel hit
	    if ( !persistentHit == 0 &&
		 typeid(*persistentHit) == typeid(SiPixelRecHit) )
	      {

		// tell the C++ compiler that the hit is a pixel hit
		const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>( hit->hit() );

		double pixel_hit_probability = (float)pixhit->clusterProbability(0);

		if ( pixel_hit_probability < 0.0 )
		  LogDebug("From KFFittingSmoother.cc")
		    << "Wraning : Negative pixel hit probability !!!! Will set the probability to 10^{-15} !!!" << endl;

		if ( pixel_hit_probability <= 0.0 || log10( pixel_hit_probability ) < log_pixel_probability_lower_limit )
		  log_pixel_hit_probability = log_pixel_probability_lower_limit;
		else
		  log_pixel_hit_probability = log10( pixel_hit_probability );

		int qbin = (int)pixhit->qBin();

		if ( ( log_pixel_hit_probability <  log_pixel_prob_cut ) &&
		     ( qbin                      != 0                  ) )
		  {
			has_low_pixel_prob = true;
			log_pixel_prob_cut = log_pixel_hit_probability;
			low_pixel_prob_Id  = tm->recHit()->geographicalId().rawId();
			low_pixel_prob_Det = tm->recHit()->det();
		  }

	      } // if ( !persistentHit == 0 && ... )

	  } // if ( hit->isValid() && ... )
	}
	// --- here the block of code about PXL Outlier Rejection ends ---

      } // for (std::vector<TrajectoryMeasurement>::const_iterator tm=vtm.begin(); tm!=vtm.end(); tm++)


      if ( hasoutliers || has_low_pixel_prob ) { // Reject outliers and pixel hits with low probability

	// Replace outlier hit or low probability pixel hit with invalid hit
	for ( unsigned int j=0; j<myHits.size(); ++j ) {
	  if ( hasoutliers && outlierId == myHits[j]->geographicalId().rawId() )
	    {
	      LogTrace("TrackFitters") << "Rejecting outlier hit  with estimate " << cut << " at position "
				       << j << " with rawId=" << myHits[j]->geographicalId().rawId();
	      LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
	      myHits[j] = InvalidTransientRecHit::build(outlierDet, TrackingRecHit::missing);
	    }
	  else if ( has_low_pixel_prob && low_pixel_prob_Id == myHits[j]->geographicalId().rawId() ){
	    LogTrace("TrackFitters") << "Rejecting low proability pixel hit with log_pixel_prob_cut = "
				     << log_pixel_prob_cut << " at position "
				     << j << " with rawId =" << myHits[j]->geographicalId().rawId();
	    LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
	    myHits[j] = InvalidTransientRecHit::build(low_pixel_prob_Det, TrackingRecHit::missing);
	  }

	} // for ( unsigned int j=0; j<myHits.size(); ++j)

	// Look if there are two consecutive invalid hits
	if ( breakTrajWith2ConsecutiveMissing ) {
	  unsigned int firstinvalid = myHits.size();
	  for ( unsigned int j=0; j<myHits.size()-1; ++j )
	    {
	      if ( ((myHits[j  ]->type() == TrackingRecHit::missing) && (myHits[j  ]->geographicalId().rawId() != 0)) &&
		   ((myHits[j+1]->type() == TrackingRecHit::missing) && (myHits[j+1]->geographicalId().rawId() != 0)) )
		{
		  firstinvalid = j;
		  LogTrace("TrackFitters") << "Found two consecutive missing hits. First invalid: " << firstinvalid;
		  break;
		}
	    }

	  //reject all the hits after the last valid before two consecutive invalid (missing) hits
	  //hits are sorted in the same order as in the track candidate FIXME??????
	  if (firstinvalid != myHits.size()) myHits.erase(myHits.begin()+firstinvalid,myHits.end());

	}

      } // if ( hasoutliers || has_low_pixel_prob )

    } // if ( theEstimateCut > 0 ... )

    if ( ( hasoutliers ||        // otherwise there won't be a 'next' loop where tmp_first could be useful
	   has_low_pixel_prob ) &&  // ggiurgiu@fnal.gov
	 !rejectTracksFlag &&  // othewrise we won't ever need tmp_first
	 firstTry ) { // only at first step
      std::swap(smoothed,tmp_first); firstTry=false;
    }

  } // do
  while ( hasoutliers || has_low_pixel_prob ); // ggiurgiu@fnal.gov

  if ( smoothed.isValid() ) {
    if ( noInvalidHitsBeginEnd )  {
      // discard latest dummy measurements
      if (!smoothed.lastMeasurement().recHitR().isValid() )
	LogTrace("TrackFitters") << "Last measurement is invalid";

      while (!smoothed.lastMeasurement().recHitR().isValid() )
	smoothed.pop();

      //remove the invalid hits at the begin of the trajectory
      if ( !smoothed.firstMeasurement().recHitR().isValid() ) {
	LogTrace("TrackFitters") << "First measurement is in`valid";
	Trajectory tmpTraj(smoothed.seed(),smoothed.direction());
	Trajectory::DataContainer  & meas = smoothed.measurements();
	auto it = meas.begin();
	for ( ; it!=meas.end(); ++it )
	  if ( it->recHitR().isValid() )  break;
	tmpTraj.push(std::move(*it),smoothed.chiSquared());//push the first valid measurement and set the same global chi2

	for (auto itt=it+1; itt!=meas.end();++itt)
	  tmpTraj.push(std::move(*itt),0);//add all the other measurements

	std::swap(smoothed,tmpTraj);

      } //  if ( !smoothed[0].firstMeasurement().recHit()->isValid() )

    } // if ( noInvalidHitsBeginEnd )

    LogTrace("TrackFitters") << "end: returning smoothed trajectory with chi2="
			     << smoothed.chiSquared() ;

    //LogTrace("TrackFitters") << "dump hits before return";
    //Trajectory::DataContainer meas = smoothed[0].measurements();
    //for (Trajectory::DataContainer::iterator it=meas.begin();it!=meas.end();++it) {
    //LogTrace("TrackFitters") << "hit #" << meas.end()-it-1 << " validity=" << it->recHit()->isValid()
    //<< " det=" << it->recHit()->geographicalId().rawId();
    //}

  }

  return smoothed;

}


Trajectory KFFittingSmoother::fitOne(const TrajectorySeed& aSeed,
				     const RecHitContainer& hits,fitType type) const{

  throw cms::Exception("TrackFitters",
		       "KFFittingSmoother::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented");

  return Trajectory();
}
