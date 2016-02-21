#include "TrackingTools/GsfTracking/interface/GsfTrajectoryFitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"


#ifdef EDM_ML_DEBUG

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

namespace {
   void dump(TrackingRecHit const & hit, int hitcounter) {
    if (hit.isValid()) {
      LogTrace("GsfTrackFitters")<< " ----------------- HIT #" << hitcounter << " (VALID)-----------------------\n"
	<< "  HIT IS AT R   " << hit.globalPosition().perp() << "\n"
	<< "  HIT IS AT Z   " << hit.globalPosition().z() << "\n"
	<< "  HIT IS AT Phi " << hit.globalPosition().phi() << "\n"
	<< "  HIT IS AT Loc " << hit.localPosition() << "\n"
	<< "  WITH LocError " << hit.localPositionError() << "\n"
	<< "  HIT IS AT Glo " << hit.globalPosition() << "\n"
	<< "SURFACE POSITION" << "\n"
	<< hit.surface()->position()<<"\n"
	<< "SURFACE ROTATION" << "\n"
	<< hit.surface()->rotation()
        <<  "dimension " << hit.dimension();

      DetId hitId = hit.geographicalId();

      LogDebug("GsfTrackFitters") << " hit det=" << hitId.rawId();

      if(hitId.det() == DetId::Tracker) {
	if (hitId.subdetId() == StripSubdetector::TIB )
	  LogDebug("GsfTrackFitters") << " I am TIB " << TIBDetId(hitId).layer();
	else if (hitId.subdetId() == StripSubdetector::TOB )
	  LogDebug("GsfTrackFitters") << " I am TOB " << TOBDetId(hitId).layer();
	else if (hitId.subdetId() == StripSubdetector::TEC )
	  LogDebug("GsfTrackFitters") << " I am TEC " << TECDetId(hitId).wheel();
	else if (hitId.subdetId() == StripSubdetector::TID )
	  LogDebug("GsfTrackFitters") << " I am TID " << TIDDetId(hitId).wheel();
	else if (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel )
	  LogDebug("GsfTrackFitters") << " I am PixBar " << PXBDetId(hitId).layer();
	else if (hitId.subdetId() == (int) PixelSubdetector::PixelEndcap )
	  LogDebug("GsfTrackFitters") << " I am PixFwd " << PXFDetId(hitId).disk();
	else
	  LogDebug("GsfTrackFitters") << " UNKNOWN TRACKER HIT TYPE ";
      }
      else if(hitId.det() == DetId::Muon) {
	if(hitId.subdetId() == MuonSubdetId::DT)
	  LogDebug("GsfTrackFitters") << " I am DT " << DTWireId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::CSC )
	  LogDebug("GsfTrackFitters") << " I am CSC " << CSCDetId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::RPC )
	  LogDebug("GsfTrackFitters") << " I am RPC " << RPCDetId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::GEM )
	  LogDebug("GsfTrackFitters") << " I am GEM " << GEMDetId(hitId);

	else if (hitId.subdetId() == MuonSubdetId::ME0 )
	  LogDebug("GsfTrackFitters") << " I am ME0 " << ME0DetId(hitId);
	else 
	  LogDebug("GsfTrackFitters") << " UNKNOWN MUON HIT TYPE ";
      }
      else
	LogDebug("GsfTrackFitters") << " UNKNOWN HIT TYPE ";

    } else {
      LogDebug("GsfTrackFitters")
	<< " ----------------- INVALID HIT #" << hitcounter << " -----------------------";
    }
   }
}
#else
namespace {
   inline void dump(TrackingRecHit const &, int) {}
}
#endif



GsfTrajectoryFitter::GsfTrajectoryFitter(const Propagator& aPropagator,
					 const TrajectoryStateUpdator& aUpdator,
					 const MeasurementEstimator& aEstimator,
					 const MultiTrajectoryStateMerger& aMerger,
					 const DetLayerGeometry* detLayerGeometry) :
  thePropagator(aPropagator.clone()),
  theUpdator(aUpdator.clone()),
  theEstimator(aEstimator.clone()),
  theMerger(aMerger.clone()),
  theGeometry(detLayerGeometry)
{
  if(!theGeometry) theGeometry = &dummyGeometry;
}

GsfTrajectoryFitter::~GsfTrajectoryFitter() {
  delete thePropagator;
  delete theUpdator;
  delete theEstimator;
  delete theMerger;
}

Trajectory GsfTrajectoryFitter::fitOne(const Trajectory& aTraj, fitType type) const {  
  if(aTraj.empty()) return Trajectory();
 
  TM const & firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());
  
  return fitOne(aTraj.seed(), aTraj.recHits(), firstTsos,type);
}

Trajectory GsfTrajectoryFitter::fitOne(const TrajectorySeed& aSeed,
				       const RecHitContainer& hits, fitType type) const {
  
  edm::LogError("GsfTrajectoryFitter") 
    << "GsfTrajectoryFitter::fit(TrajectorySeed, vector<RecHit>) not implemented";
  
  return Trajectory();
}

Trajectory GsfTrajectoryFitter::fitOne(const TrajectorySeed& aSeed,
				    const RecHitContainer& hits, 
				    const TrajectoryStateOnSurface& firstPredTsos,
				    fitType) const {

  if(hits.empty()) return Trajectory();

  Trajectory myTraj(aSeed, propagator()->propagationDirection());

  TSOS predTsos(firstPredTsos);
  if(!predTsos.isValid()) {
    edm::LogInfo("GsfTrackFitter") 
      << "GsfTrajectoryFitter: predicted tsos of first measurement not valid!";
    return Trajectory();
  } 

  TSOS currTsos;
  if(hits.front()->isValid()) {
     auto const & ihit = hits.front();
    //update
     assert( (!(ihit)->canImproveWithTrack()) | (nullptr!=theHitCloner));
     assert( (!(ihit)->canImproveWithTrack()) | (nullptr!=dynamic_cast<BaseTrackerRecHit const*>(ihit.get())));
     auto preciseHit = theHitCloner->makeShared(ihit,predTsos);
     dump(*preciseHit,1);
    {
      currTsos = updator()->update(predTsos, *preciseHit);
    }
    if (!predTsos.isValid() || !currTsos.isValid()){
      edm::LogError("InvalidState")<<"first hit";
      return Trajectory();
    }
    myTraj.push(TM(predTsos, currTsos, preciseHit, 0., theGeometry->idToLayer(preciseHit->geographicalId() )),
		estimator()->estimate(predTsos, *preciseHit).second);
  } else {
    currTsos = predTsos;
    if (!predTsos.isValid()){
      edm::LogError("InvalidState")<<"first invalid hit";
      return Trajectory();
    }
    myTraj.push(TM(predTsos, *hits.begin(),0., theGeometry->idToLayer((*hits.begin())->geographicalId()) ));
  }
  
  int hitcounter = 1;
  for(RecHitContainer::const_iterator ihit = hits.begin() + 1; 
      ihit != hits.end(); ihit++) {
        ++hitcounter;
    
    //
    // temporary protection copied from KFTrajectoryFitter.
    //
    if ((**ihit).isValid() == false && (**ihit).det() == 0) {
      LogDebug("GsfTrajectoryFitter") << " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }

    {
      //       TimeMe t(*propTimer,false);
      predTsos = propagator()->propagate(currTsos,
					 (**ihit).det()->surface());
    }
    if(!predTsos.isValid()) {
      if ( myTraj.foundHits()>=3 ) {
	edm::LogInfo("GsfTrackFitter") 
	  << "GsfTrajectoryFitter: predicted tsos not valid! \n"
	  << "Returning trajectory with " << myTraj.foundHits() << " found hits.";
	return myTraj;
      }
      else {
      edm::LogInfo("GsfTrackFitter") 
	<< "GsfTrajectoryFitter: predicted tsos not valid after " << myTraj.foundHits()
	<< " hits, discarding candidate!";
	return Trajectory();
      }
    }
    if ( merger() ) predTsos = merger()->merge(predTsos);
    
    if((**ihit).isValid()) {
      //update
       assert( (!(*ihit)->canImproveWithTrack()) | (nullptr!=theHitCloner));
       assert( (!(*ihit)->canImproveWithTrack()) | (nullptr!=dynamic_cast<BaseTrackerRecHit const*>((*ihit).get())));
       auto preciseHit = theHitCloner->makeShared(*ihit,predTsos);
       dump(*preciseHit,hitcounter);
      {
	currTsos = updator()->update(predTsos, *preciseHit);
      }
      if (!predTsos.isValid() || !currTsos.isValid()){
	edm::LogError("InvalidState")<<"inside hit";
	return Trajectory();
      }
      myTraj.push(TM(predTsos, currTsos, preciseHit,
		     estimator()->estimate(predTsos, *preciseHit).second,
		     theGeometry->idToLayer(preciseHit->geographicalId() )));
    } else {
      currTsos = predTsos;
      if (!predTsos.isValid()){
      edm::LogError("InvalidState")<<"inside invalid hit";
      return Trajectory();
      }
      myTraj.push(TM(predTsos, *ihit,0., theGeometry->idToLayer( (*ihit)->geographicalId()) ));
    }
    LogTrace("GsfTrackFitters")
      << "predTsos !" << "\n"
      << predTsos 
      <<" with local position " << predTsos.localPosition()
      <<"currTsos !" << "\n"
      << currTsos
      <<" with local position " << currTsos.localPosition();
  }
  return myTraj;
}
