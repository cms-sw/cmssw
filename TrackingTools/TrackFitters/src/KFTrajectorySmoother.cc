#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"

#ifdef EDM_ML_DEBUG
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#endif

#include "FWCore/Utilities/interface/GCC11Compatibility.h"


KFTrajectorySmoother::~KFTrajectorySmoother() {

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}

Trajectory
KFTrajectorySmoother::trajectory(const Trajectory& aTraj) const {

  if(aTraj.empty()) return Trajectory();

  if (  aTraj.direction() == alongMomentum) {
    thePropagator->setPropagationDirection(oppositeToMomentum);
  } else {
    thePropagator->setPropagationDirection(alongMomentum);
  }
  
  const std::vector<TM> & avtm = aTraj.measurements();
  
  Trajectory ret(aTraj.seed(), thePropagator->propagationDirection());
  Trajectory & myTraj = ret;
  myTraj.reserve(avtm.size());
  
  
  
#ifdef EDM_ML_DEBUG
  LogDebug("TrackFitters") << "KFTrajectorySmoother::trajectories starting with " << avtm.size() << " HITS\n";
  for (unsigned int j=0;j<avtm.size();j++) { 
    if (avtm[j].recHit()->det()) 
      LogTrace("TrackFitters") << "hit #:" << j+1 << " rawId=" << avtm[j].recHit()->det()->geographicalId().rawId() 
			       << " validity=" << avtm[j].recHit()->isValid();
    else
      LogTrace("TrackFitters") << "hit #:" << j+1 << " Hit with no Det information";
  }
#endif // EDM_ML_DEBUG
  
  
  TSOS predTsos = avtm.back().forwardPredictedState();
  predTsos.rescaleError(theErrorRescaling);
  TSOS currTsos;
  
  TrajectoryStateCombiner combiner;
  
  unsigned int hitcounter = avtm.size();
  for(std::vector<TM>::const_reverse_iterator itm = avtm.rbegin(); itm != (avtm.rend()); ++itm,--hitcounter) {

    TransientTrackingRecHit::ConstRecHitPointer hit = itm->recHit();

    //check surface just for safety: should never be ==0 because they are skipped in the fitter 
    // if unlikely(hit->det() == nullptr) continue;
    if unlikely( hit->surface()==nullptr ) {
	LogDebug("TrackFitters")<< " Error: invalid hit with no GeomDet attached .... skipping";
	continue;
      }

   if (hit->det() && hit->geographicalId()<1000U) std::cout << "Problem 0 det id for " << typeid(*hit).name() << ' ' <<  hit->det()->geographicalId()  << std::endl;
   if (hit->isValid() && hit->geographicalId()<1000U) std::cout << "Problem 0 det id for " << typeid(*hit).name() << ' ' <<  hit->det()->geographicalId()  << std::endl;


    if (hitcounter != avtm.size())//no propagation needed for first smoothed (==last fitted) hit 
      predTsos = thePropagator->propagate( currTsos, *(hit->surface()) );

    if unlikely(!predTsos.isValid()) {
	LogDebug("TrackFitters") << "KFTrajectorySmoother: predicted tsos not valid!";
	if( myTraj.foundHits() >= minHits_ ) {
	  LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	} else {        
	  LogDebug("TrackFitters") << " killing trajectory" << "\n";      
	  return Trajectory();
	}
	break;      
      }

    if(hit->isValid()) {
 
#ifdef EDM_ML_DEBUG
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
      
      DetId hitId = hit->geographicalId();
      
      if(hitId.det() == DetId::Tracker) {
	if (hitId.subdetId() == StripSubdetector::TIB )  
	  LogTrace("TrackFitters") << " I am TIB " << TIBDetId(hitId).layer();
	else if (hitId.subdetId() == StripSubdetector::TOB ) 
	  LogTrace("TrackFitters") << " I am TOB " << TOBDetId(hitId).layer();
	else if (hitId.subdetId() == StripSubdetector::TEC ) 
	  LogTrace("TrackFitters") << " I am TEC " << TECDetId(hitId).wheel();
	else if (hitId.subdetId() == StripSubdetector::TID ) 
	  LogTrace("TrackFitters") << " I am TID " << TIDDetId(hitId).wheel();
	else if (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel ) 
	  LogTrace("TrackFitters") << " I am PixBar " << PXBDetId(hitId).layer();
	else if (hitId.subdetId() == (int) PixelSubdetector::PixelEndcap )
	  LogTrace("TrackFitters") << " I am PixFwd " << PXFDetId(hitId).disk();
	else 
	  LogTrace("TrackFitters") << " UNKNOWN TRACKER HIT TYPE ";
      }
      else if(hitId.det() == DetId::Muon) {
	if(hitId.subdetId() == MuonSubdetId::DT)
	  LogTrace("TrackFitters") << " I am DT " << DTWireId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::CSC )
	  LogTrace("TrackFitters") << " I am CSC " << CSCDetId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::RPC )
	  LogTrace("TrackFitters") << " I am RPC " << RPCDetId(hitId);
	else 
	  LogTrace("TrackFitters") << " UNKNOWN MUON HIT TYPE ";
      }
      else
	LogTrace("TrackFitters") << " UNKNOWN HIT TYPE ";
#endif //EDM_ML_DEBUG
      
      
      
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
      
      if unlikely(!combTsos.isValid()) {
	  LogDebug("TrackFitters") << 
	    "KFTrajectorySmoother: combined tsos not valid!\n" <<
	    "pred Tsos pos: " << predTsos.globalPosition() << "\n" <<
	    "pred Tsos mom: " << predTsos.globalMomentum() << "\n" <<
	    "TrackingRecHit: " << hit->surface()->toGlobal(hit->localPosition()) << "\n" ;
	  if( myTraj.foundHits() >= minHits_ ) {
	    LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	  } else {        
	    LogDebug("TrackFitters") << " killing trajectory" << "\n";       
	    return Trajectory();
	  }
	  break;      
	}
      
        assert(hit->geographicalId()!=0U);
       	assert(hit->surface()!=nullptr);
        assert( (!(hit)->canImproveWithTrack()) | (nullptr!=theHitCloner));
        assert( (!(hit)->canImproveWithTrack()) | (nullptr!=dynamic_cast<BaseTrackerRecHit const*>(hit.get())));
        auto preciseHit = theHitCloner->makeShared(hit,combTsos);
        assert(preciseHit->isValid());
       	assert(preciseHit->geographicalId()!=0U);
       	assert(preciseHit->surface()!=nullptr);

      if unlikely(!preciseHit->isValid()){
	  LogTrace("TrackFitters") << "THE Precise HIT IS NOT VALID: using currTsos = predTsos" << "\n";
	  currTsos = predTsos;
	  myTraj.push(TM(predTsos, hit, 0, theGeometry->idToLayer(hit->geographicalId()) ));
	}else{
	LogTrace("TrackFitters") << "THE Precise HIT IS VALID: updating currTsos" << "\n";
	
	//update backward predicted tsos with the hit
	currTsos = updator()->update(predTsos, *preciseHit);
        if unlikely(!currTsos.isValid()) {
	    currTsos = predTsos;
	    edm::LogWarning("KFSmoother_UpdateFailed") << 
	      "Failed updating state with hit. Rolling back to non-updated state.\n" <<
	      "State: "   << predTsos << 
	      "Hit local pos:  " << hit->localPosition() << "\n" <<
	      "Hit local err:  " << hit->localPositionError() << "\n" <<
	      "Hit global pos: " << hit->globalPosition() << "\n" <<
	      "Hit global err: " << hit->globalPositionError().matrix() << 
	      "\n";
	  }
	
	//smooTsos updates the N-1 hits prediction with the hit
	if (hitcounter == avtm.size()) smooTsos = itm->updatedState();
	else if (hitcounter == 1) smooTsos = currTsos;
	else smooTsos = combiner(itm->forwardPredictedState(), currTsos); 
	
	if unlikely(!smooTsos.isValid()) {
	    LogDebug("TrackFitters") << "KFTrajectorySmoother: smoothed tsos not valid!";
	    if( myTraj.foundHits() >= minHits_ ) {
	      LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	    } else {        
	      LogDebug("TrackFitters") << " killing trajectory" << "\n";       
	      return Trajectory();  
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
	
	//check for valid hits with no det (refitter with constraints)
	if (preciseHit->det()) myTraj.push(TM(itm->forwardPredictedState(),
					      predTsos,
					      smooTsos,
					      preciseHit,
					      estimate,
					      theGeometry->idToLayer(preciseHit->geographicalId()) ),
					   estimator()->estimate(predTsos,*preciseHit).second);
	else myTraj.push(TM(itm->forwardPredictedState(),
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
      
      if unlikely(!combTsos.isValid()) {
    	LogDebug("TrackFitters") << 
    	  "KFTrajectorySmoother: combined tsos not valid!";
        return Trajectory();
	}
      assert( (hit->det()==nullptr) || hit->geographicalId()!=0U);
      if (hit->det()) 
         myTraj.push(TM(itm->forwardPredictedState(),
		     predTsos,
    		     combTsos,
    		     hit,
		     0,
		     theGeometry->idToLayer(hit->geographicalId()) ));
     else myTraj.push(TM(itm->forwardPredictedState(),
                     predTsos,
                     combTsos,
                     hit,
                     0));

    }
  } // for loop
  
  return ret; 
  
}
