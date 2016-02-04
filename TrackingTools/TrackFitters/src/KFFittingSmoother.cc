#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
// #include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TransientTrackingRecHit/interface/InvalidTransientRecHit.h"

// ggiurgiu@fnal.gov: Add headers needed to cut on pixel hit probability
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

using namespace std;

KFFittingSmoother::~KFFittingSmoother() 
{
  delete theSmoother;
  delete theFitter;
}

vector<Trajectory> KFFittingSmoother::fit(const Trajectory& t) const 
{
  vector<Trajectory> smoothed;
  if ( t.isValid() ) 
    { 
      vector<Trajectory> fitted = fitter()->fit(t);
      smoothingStep(fitted, smoothed);
    }
  return smoothed;
}

vector<Trajectory> KFFittingSmoother::fit(const TrajectorySeed& aSeed,
					  const RecHitContainer& hits, 
					  const TrajectoryStateOnSurface& firstPredTsos) const 
{
  LogDebug("TrackFitters") << "In KFFittingSmoother::fit";

  //if(hits.empty()) return vector<Trajectory>(); // gio: moved later to optimize return value
  
  bool hasoutliers;
  bool has_low_pixel_prob; // ggiurgiu@fnal.gov: Add flag for pixel hits with low template probability

  // ggiurgiu@fnal.gov: If log(Prob) < -15.0 or if Prob <= 0.0 then set log(Prob) = -15.0
  double log_pixel_probability_lower_limit = -15.0;

  RecHitContainer myHits = hits; 
  vector<Trajectory> smoothed;
  vector<Trajectory> tmp_first;

  do
    {
      if ( hits.empty() ) 
	{ 
	  smoothed.clear(); 
	  break; 
	}
      
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
      vector<Trajectory> fitted = fitter()->fit(aSeed, myHits, firstPredTsos);
      //call the smoother
      smoothed.clear();
      smoothingStep(fitted, smoothed);
      
      //if (tmp_first.size()==0) tmp_first = smoothed; moved later
      
      if ( smoothed.empty() ) 
	{
	  if ( rejectTracksFlag )
	    {
	      LogTrace("TrackFitters") << "smoothed.size()==0 => trajectory rejected";
	      //return vector<Trajectory>(); // break is enough to get this
	    } 
	  else 
	    {
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
      
      if ( theEstimateCut > 0 || log_pixel_prob_cut > log_pixel_probability_lower_limit ) 
	{
	  if ( smoothed[0].foundHits() < theMinNumberOfHits ) 
	    {
	      if ( rejectTracksFlag ) 
		{
		  LogTrace("TrackFitters") << "smoothed[0].foundHits()<theMinNumberOfHits => trajectory rejected";
		  smoothed.clear();
		  //return vector<Trajectory>(); // break is enough
		} 
	      else 
		{
		  // it might be it's the first step
		  if ( !tmp_first.empty() ) 
		    { 
		      tmp_first.swap(smoothed); 
		    } 
		  
		  LogTrace("TrackFitters") 
		    << "smoothed[0].foundHits()<theMinNumberOfHits => returning orignal trajectory with chi2=" 
		    <<  smoothed[0].chiSquared() ;
		}
	      break;
	    }
	  
	  // Check if there are outliers or low probability pixel rec hits
	  const std::vector<TrajectoryMeasurement> & vtm = smoothed[0].measurements();
	  
	  double log_pixel_hit_probability = -999999.9;
	  
	  for (std::vector<TrajectoryMeasurement>::const_iterator tm=vtm.begin(); tm!=vtm.end();tm++)
	    {
	      double estimate = tm->estimate();
	      
	      // --- here is the block of code about generic chi2-based Outlier Rejection ---
	      if ( estimate > cut ) 
		{
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
		     )
		  {
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
	  
      
	  if ( hasoutliers || has_low_pixel_prob ) 
	    { // Reject outliers and pixel hits with low probability 
	      
	      // Replace outlier hit or low probability pixel hit with invalid hit
	      for ( unsigned int j=0; j<myHits.size(); ++j ) 
		{ 
		  if ( hasoutliers && outlierId == myHits[j]->geographicalId().rawId() )
		    {
		      LogTrace("TrackFitters") << "Rejecting outlier hit  with estimate " << cut << " at position " 
					       << j << " with rawId=" << myHits[j]->geographicalId().rawId();
		      LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
		      myHits[j] = InvalidTransientRecHit::build(outlierDet, TrackingRecHit::missing);
		    }
		  else if ( has_low_pixel_prob && low_pixel_prob_Id == myHits[j]->geographicalId().rawId() )
		    {
		      LogTrace("TrackFitters") << "Rejecting low proability pixel hit with log_pixel_prob_cut = " 
					       << log_pixel_prob_cut << " at position " 
					       << j << " with rawId =" << myHits[j]->geographicalId().rawId();
		      LogTrace("TrackFitters") << "The fit will be repeated without the outlier";
		      myHits[j] = InvalidTransientRecHit::build(low_pixel_prob_Det, TrackingRecHit::missing);
		    }
		  
		} // for ( unsigned int j=0; j<myHits.size(); ++j)
	      
	      // Look if there are two consecutive invalid hits
	      if ( breakTrajWith2ConsecutiveMissing ) 
		{
		  unsigned int firstinvalid = myHits.size()-1;
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
		  myHits.erase(myHits.begin()+firstinvalid,myHits.end());
		  
		}
	      
	    } // if ( hasoutliers || has_low_pixel_prob ) 
	  
	} // if ( theEstimateCut > 0 ... )
  
      if ( ( hasoutliers ||        // otherwise there won't be a 'next' loop where tmp_first could be useful 
	     has_low_pixel_prob ) &&  // ggiurgiu@fnal.gov
	   !rejectTracksFlag &&  // othewrise we won't ever need tmp_first
	   tmp_first.empty() ) 
	{ // only at first step
	  smoothed.swap(tmp_first);
	}   
      
    } // do
  while ( hasoutliers || has_low_pixel_prob ); // ggiurgiu@fnal.gov
  
  if ( !smoothed.empty() ) 
    {
      if ( noInvalidHitsBeginEnd ) 
	{
	  // discard latest dummy measurements
	  if ( !smoothed[0].empty() && 
	       !smoothed[0].lastMeasurement().recHit()->isValid() ) 
	    LogTrace("TrackFitters") << "Last measurement is invalid";
	
	  while ( !smoothed[0].empty() && 
		  !smoothed[0].lastMeasurement().recHit()->isValid() ) 
	    smoothed[0].pop();
	  
	  //remove the invalid hits at the begin of the trajectory
	  if ( !smoothed[0].firstMeasurement().recHit()->isValid() ) 
	    {
	      LogTrace("TrackFitters") << "First measurement is invalid";
	      Trajectory tmpTraj(smoothed[0].seed(),smoothed[0].direction());
	      Trajectory::DataContainer meas = smoothed[0].measurements();
	      
	      Trajectory::DataContainer::iterator it;//first valid hit
	      for ( it=meas.begin(); it!=meas.end(); ++it ) 
		{
		  if ( !it->recHit()->isValid() ) 
		    continue;
		  else break;
		}
	      tmpTraj.push(*it,smoothed[0].chiSquared());//push the first valid measurement and set the same global chi2
	     
	      for (Trajectory::DataContainer::iterator itt=it+1; itt!=meas.end();++itt) 
		{
		  tmpTraj.push(*itt,0);//add all the other measurements
		}
	      
	      smoothed.clear();
	      smoothed.push_back(tmpTraj);
	   
	    } //  if ( !smoothed[0].firstMeasurement().recHit()->isValid() ) 
	
	} // if ( noInvalidHitsBeginEnd ) 
      
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
