#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

KFTrajectoryFitter::~KFTrajectoryFitter() {

  delete thePropagator;
  delete theUpdator;
  delete theEstimator;

}


std::vector<Trajectory> KFTrajectoryFitter::fit(const Trajectory& aTraj) const {

  if(aTraj.empty()) return std::vector<Trajectory>();
 
  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fit(aTraj.seed(), aTraj.recHits(), firstTsos);
}

std::vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
						const RecHitContainer& hits) const{

  throw cms::Exception("TrackFitters", 
		       "KFTrajectoryFitter::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented"); 

  return std::vector<Trajectory>();
}

std::vector<Trajectory> KFTrajectoryFitter::fit(const TrajectorySeed& aSeed,
						const RecHitContainer& hits,
						const TSOS& firstPredTsos) const 
{
  if(hits.empty()) return std::vector<Trajectory>();

  if (aSeed.direction() == alongMomentum) {
    thePropagator->setPropagationDirection(alongMomentum);
  } else if (aSeed.direction() == oppositeToMomentum){
    thePropagator->setPropagationDirection(oppositeToMomentum);
  } else {
    throw cms::Exception("KFTrajectoryFitter","TrajectorySeed::direction() requested but not set");
  }

  LogDebug("TrackFitters")
    <<" +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    <<" KFTrajectoryFitter::fit starting with " << hits.size() <<" HITS";
  
  for (unsigned int j=0;j<hits.size();j++) { 
    if (hits[j]->det()) 
      LogTrace("TrackFitters") << "hit #:" << j+1 << " rawId=" << hits[j]->det()->geographicalId().rawId() 
			       << " validity=" << hits[j]->isValid();
    else
      LogTrace("TrackFitters") << "hit #:" << j+1 << " Hit with no Det information";
  }
  LogTrace("TrackFitters") << " INITIAL STATE "<< firstPredTsos;

  std::vector<Trajectory> ret(1, Trajectory(aSeed, thePropagator->propagationDirection()));
  Trajectory & myTraj = ret.front();
  myTraj.reserve(hits.size());

  TSOS predTsos(firstPredTsos);
  TSOS currTsos;

  int hitcounter = 1;
  for(RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end(); ++ihit, ++hitcounter) {

    const TransientTrackingRecHit & hit = (**ihit);

    if (hit.isValid() == false && hit.surface() == 0) {
      LogDebug("TrackFitters")<< " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }

    if (hit.isValid()) {
      LogTrace("TrackFitters")
	<< " ----------------- HIT #" << hitcounter << " (VALID)-----------------------\n"
	<< "  HIT IS AT R   " << hit.globalPosition().perp() << "\n"
	<< "  HIT IS AT Z   " << hit.globalPosition().z() << "\n"
	<< "  HIT IS AT Phi " << hit.globalPosition().phi() << "\n"
	<< "  HIT IS AT Loc " << hit.localPosition() << "\n"
	<< "  WITH LocError " << hit.localPositionError() << "\n"
	<< "  HIT IS AT Glo " << hit.globalPosition() << "\n"
	<< "SURFACE POSITION" << "\n"
	<< hit.surface()->position()<<"\n"
	<< "SURFACE ROTATION" << "\n"
	<< hit.surface()->rotation();
      LogTrace("TrackFitters") << " hit det=" << hit.geographicalId().rawId();
      if (hit.geographicalId().subdetId() == StripSubdetector::TIB ) {
	LogTrace("TrackFitters") << " I am TIB " << TIBDetId(hit.geographicalId()).layer();
      } else if (hit.geographicalId().subdetId() == StripSubdetector::TOB ) { 
	LogTrace("TrackFitters") << " I am TOB " << TOBDetId(hit.geographicalId()).layer();
      } else if (hit.geographicalId().subdetId() == StripSubdetector::TEC ) { 
	LogTrace("TrackFitters") << " I am TEC " << TECDetId(hit.geographicalId()).wheel();
      } else if (hit.geographicalId().subdetId() == StripSubdetector::TID ) { 
	LogTrace("TrackFitters") << " I am TID " << TIDDetId(hit.geographicalId()).wheel();
      } else if (hit.geographicalId().subdetId() == StripSubdetector::TID ) { 
	LogTrace("TrackFitters") << " I am TID " << TIDDetId(hit.geographicalId()).wheel();
      } else if (hit.geographicalId().subdetId() == (int) PixelSubdetector::PixelBarrel ) {
	LogTrace("TrackFitters") << " I am PixBar " << PXBDetId(hit.geographicalId()).layer();
      } else if (hit.geographicalId().subdetId() == (int) PixelSubdetector::PixelEndcap ){
	LogTrace("TrackFitters") << " I am PixFwd " << PXFDetId(hit.geographicalId()).disk();
      } else {
	LogTrace("TrackFitters") << " UNKNOWN HIT TYPE ";
      }
    } else {
      LogTrace("TrackFitters")
	<< " ----------------- HIT #" << hitcounter << " (INVALID)-----------------------";      
    }

    if ( hitcounter != 1) //no propagation needed for the first hit
      predTsos = thePropagator->propagate( currTsos, *(hit.surface()) );
    
    if(!predTsos.isValid()) {
      LogDebug("TrackFitters") 
	<< "SOMETHING WRONG !" << "\n"
	<< "KFTrajectoryFitter: predicted tsos not valid!\n" 
	<< "current TSOS: " << currTsos << "\n";

      if(hit.surface())	LogTrace("TrackFitters") << "next Surface: " << hit.surface()->position() << "\n";

      if( myTraj.foundHits() >= minHits_ ) {
	LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	break;      
      } else {        
	LogDebug("TrackFitters") << " killing trajectory" << "\n";       
	return std::vector<Trajectory>();
      }
    }
    
    if(hit.isValid()) {
      //update
      LogTrace("TrackFitters") << "THE HIT IS VALID: updating hit with predTsos";
      TransientTrackingRecHit::RecHitPointer preciseHit = hit.clone(predTsos);

      if (preciseHit->isValid() == false){
	LogTrace("TrackFitters") << "THE Precise HIT IS NOT VALID: using currTsos = predTsos" << "\n";
	currTsos = predTsos;
	myTraj.push(TM(predTsos, *ihit ));//why no estimate? if the hit is valid it should contribute to chi2...

      }else{
	LogTrace("TrackFitters") << "THE Precise HIT IS VALID: updating currTsos" << "\n";
	currTsos = updator()->update(predTsos, *preciseHit);
	myTraj.push(TM(predTsos, currTsos, preciseHit,
		       estimator()->estimate(predTsos, *preciseHit).second));

      }
    } else {
      //no update
      LogDebug("TrackFitters") << "THE HIT IS NOT VALID: using currTsos" << "\n";
      currTsos = predTsos;
      myTraj.push(TM(predTsos, *ihit));
    }

    LogTrace("TrackFitters")
      << "predTsos !" << "\n"
      << predTsos << "\n"
      <<"currTsos !" << "\n"
      << currTsos;
  }  
  LogDebug("TrackFitters") << "Found 1 trajectory with " << myTraj.foundHits() << " valid hits\n";
  
  return ret;
}

