#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

KFTrajectorySmoother::~KFTrajectorySmoother() {

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}

std::vector<Trajectory> 
KFTrajectorySmoother::trajectories(const Trajectory& aTraj) const {
  // let's try to get return value optimization
  // the 'standard' case is when we return 1 tractory

  if (  aTraj.direction() == alongMomentum) {
    thePropagator->setPropagationDirection(oppositeToMomentum);
  } else {
    thePropagator->setPropagationDirection(alongMomentum);
  }

  std::vector<Trajectory> ret(1, Trajectory(aTraj.seed(), thePropagator->propagationDirection()));
  Trajectory & myTraj = ret.front();


  if(aTraj.empty()) { ret.clear(); return ret; } 


  const std::vector<TM> & avtm = aTraj.measurements();
  LogDebug("TrackFitters") << "KFTrajectorySmoother::trajectories starting with " << avtm.size() << " HITS\n";

  myTraj.reserve(avtm.size());

  for (unsigned int j=0;j<avtm.size();j++) { 
    if (avtm[j].recHit()->det()) 
      LogTrace("TrackFitters") << "hit #:" << j+1 << " rawId=" << avtm[j].recHit()->det()->geographicalId().rawId() 
			       << " validity=" << avtm[j].recHit()->isValid();
    else
      LogTrace("TrackFitters") << "hit #:" << j+1 << " Hit with no Det information";
  }

  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);
  TSOS currTsos;
 
  TrajectoryStateCombiner combiner;

  unsigned int hitcounter = avtm.size();
  for(std::vector<TM>::const_reverse_iterator itm = avtm.rbegin(); itm != (avtm.rend()); ++itm,--hitcounter) {

    TransientTrackingRecHit::ConstRecHitPointer hit = itm->recHit();

    //check surface just for safety: should never be ==0 because they are skipped in the fitter 
    if ( hit->surface()==0 ) {
      LogDebug("TrackFitters")<< " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }

    if (hitcounter != avtm.size())//no propagation needed for first smoothed (==last fitted) hit 
      predTsos = thePropagator->propagate( currTsos, *(hit->surface()) );

    if(!predTsos.isValid()) {
      LogDebug("TrackFitters") << "KFTrajectorySmoother: predicted tsos not valid!";
      if( myTraj.foundHits() >= minHits_ ) {
	LogDebug("TrackFitters") << " breaking trajectory" << "\n";
      } else {        
	LogDebug("TrackFitters") << " killing trajectory" << "\n";      
        ret.clear(); 
      }
      break;      
    }

    if(hit->isValid()) {
      LogDebug("TrackFitters")
	<< "----------------- HIT #" << hitcounter << " (VALID)-----------------------\n"
	<< "HIT IS AT R   " << hit->globalPosition().perp() << "\n"
	<< "HIT IS AT Z   " << hit->globalPosition().z() << "\n"
	<< "HIT IS AT Phi " << hit->globalPosition().phi() << "\n"
	<< "HIT IS AT Loc " << hit->localPosition() << "\n"
	<< "WITH LocError " << hit->localPositionError() << "\n"
	<< "HIT IS AT Glo " << hit->globalPosition() << "\n"
	<< "SURFACE POSITION: " << hit->surface()->position() << "\n"
	<< "SURFACE ROTATION: " << hit->surface()->rotation() << "\n"
	<< "hit geographicalId=" << hit->geographicalId().rawId();
      if (hit->geographicalId().subdetId() == StripSubdetector::TIB ) {
	LogTrace("TrackFitters") << "I am TIB " << TIBDetId(hit->geographicalId()).layer();
      }else if (hit->geographicalId().subdetId() == StripSubdetector::TOB ) { 
	LogTrace("TrackFitters") << "I am TOB "<< TOBDetId(hit->geographicalId()).layer();
      }else if (hit->geographicalId().subdetId() == StripSubdetector::TEC ) { 
	LogTrace("TrackFitters") << "I am TEC "<< TECDetId(hit->geographicalId()).wheel();
      }else if (hit->geographicalId().subdetId() == StripSubdetector::TID ) { 
	LogTrace("TrackFitters") << "I am TID "<< TIDDetId(hit->geographicalId()).wheel();
      }else if (hit->geographicalId().subdetId() == (int) PixelSubdetector::PixelBarrel ) {
	LogTrace("TrackFitters") << "I am PixBar "<< PXBDetId(hit->geographicalId()).layer();
      } else if (hit->geographicalId().subdetId() == (int) PixelSubdetector::PixelEndcap ){
	LogTrace("TrackFitters") << "I am PixFwd " << PXFDetId(hit->geographicalId()).disk();
      } else {
	LogTrace("TrackFitters") << "UNKNOWN HIT TYPE";
      }
    
      TSOS combTsos,smooTsos;

      //3 different possibilities to calculate smoothed state:
      //1: update combined predictions with hit
      //2: combine fwd-prediction with bwd-filter
      //3: combine bwd-prediction with fwd-filter

      //combTsos is the predicted state with N-1 hits information. this means: 
      //forward predicted state for first smoothed (last fitted) hit
      //backward predicted state for last smoothed (first fitted) hit
      //combination of forward and backward predictions for other hits
      if (hitcounter == avtm.size()) combTsos = itm->forwardPredictedState();
      else if (hitcounter == 1) combTsos = predTsos;
      else combTsos = combiner(predTsos, itm->forwardPredictedState());
      if(!combTsos.isValid()) {
	LogDebug("TrackFitters") << 
	  "KFTrajectorySmoother: combined tsos not valid!\n" <<
	  "pred Tsos pos: " << predTsos.globalPosition() << "\n" <<
	  "pred Tsos mom: " << predTsos.globalMomentum() << "\n" <<
	  "TrackingRecHit: " << hit->surface()->toGlobal(hit->localPosition()) << "\n" ;
	if( myTraj.foundHits() >= minHits_ ) {
	  LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	} else {        
	  LogDebug("TrackFitters") << " killing trajectory" << "\n";       
          ret.clear(); 
        }
        break;      
      }

      TransientTrackingRecHit::RecHitPointer preciseHit = hit->clone(combTsos);

      if (preciseHit->isValid() == false){
	LogTrace("TrackFitters") << "THE Precise HIT IS NOT VALID: using currTsos = predTsos" << "\n";
	currTsos = predTsos;
	myTraj.push(TM(predTsos, hit ));//why no estimate? if the hit is valid it should contribute to chi2...
      }else{
	LogTrace("TrackFitters") << "THE Precise HIT IS VALID: updating currTsos" << "\n";
	
	//update backward predicted tsos with the hit
	currTsos = updator()->update(predTsos, *preciseHit);

	//smooTsos updates the N-1 hits prediction with the hit
	if (hitcounter == avtm.size()) smooTsos = itm->updatedState();
	else if (hitcounter == 1) smooTsos = currTsos;
	else smooTsos = combiner(itm->forwardPredictedState(), currTsos); 
	
	if(!smooTsos.isValid()) {
	  LogDebug("TrackFitters") << "KFTrajectorySmoother: smoothed tsos not valid!";
	  if( myTraj.foundHits() >= minHits_ ) {
	    LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	  } else {        
	    LogDebug("TrackFitters") << " killing trajectory" << "\n";       
            ret.clear(); 
	  }
          break;
	}
	
	double estimate;
	if (hitcounter != avtm.size()) estimate = estimator()->estimate(combTsos, *preciseHit ).second;//correct?
	else estimate = itm->estimate();
	
	LogTrace("TrackFitters")
	  << "predTsos !" << "\n"
	  << predTsos << "\n"
	  << "currTsos !" << "\n"
	  << currTsos << "\n"
	  << "smooTsos !" << "\n"
	  << smooTsos << "\n"
	  << "smoothing estimate (with combTSOS)=" << estimate << "\n"
	  << "filtering estimate=" << itm->estimate() << "\n";
	
	myTraj.push(TM(itm->forwardPredictedState(),
		       predTsos,
		       smooTsos,
		       preciseHit,
		       estimate),
		    estimator()->estimate(predTsos,*preciseHit).second);
	            //itm->estimate());
      }
    } else {
      LogDebug("TrackFitters") 
	<< "----------------- HIT #" << hitcounter << " (INVALID)-----------------------";      

      //no update
      currTsos = predTsos;
      TSOS combTsos;
      if (hitcounter == avtm.size()) combTsos = itm->forwardPredictedState();
      else if (hitcounter == 1) combTsos = predTsos;
      else combTsos = combiner(predTsos, itm->forwardPredictedState());
      
      if(!combTsos.isValid()) {
    	LogDebug("TrackFitters") << 
    	  "KFTrajectorySmoother: combined tsos not valid!";
        ret.clear(); break;
      }
      
      myTraj.push(TM(itm->forwardPredictedState(),
    		     predTsos,
    		     combTsos,
    		     hit));
    }
  } // for loop

  return ret; 
  
}
