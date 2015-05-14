#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/TkCloner.h"

#ifdef EDM_ML_DEBUG
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

const DetLayerGeometry KFTrajectoryFitter::dummyGeometry;

Trajectory KFTrajectoryFitter::fitOne(const Trajectory& aTraj, fitType type) const {

  if(aTraj.empty()) return Trajectory();

  TM firstTM = aTraj.firstMeasurement();
  TSOS firstTsos = TrajectoryStateWithArbitraryError()(firstTM.updatedState());

  return fitOne(aTraj.seed(), aTraj.recHits(), firstTsos,type);
}

Trajectory KFTrajectoryFitter::fitOne(const TrajectorySeed&,
				      const RecHitContainer&, fitType) const{

  throw cms::Exception("TrackFitters",
		       "KFTrajectoryFitter::fit(TrajectorySeed, <TransientTrackingRecHit>) not implemented");

  return Trajectory();
}

Trajectory KFTrajectoryFitter::fitOne(const TrajectorySeed& aSeed,
				      const RecHitContainer& hits,
				      const TSOS& firstPredTsos,fitType) const
{
  if(hits.empty()) return Trajectory();


  if unlikely(aSeed.direction() == anyDirection)
    throw cms::Exception("KFTrajectoryFitter","TrajectorySeed::direction() requested but not set");

  std::unique_ptr<Propagator> p_cloned = SetPropagationDirection(*thePropagator,
                                                                 aSeed.direction());

#ifdef EDM_ML_DEBUG
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
#endif

  Trajectory ret(aSeed, p_cloned->propagationDirection());
  Trajectory & myTraj = ret;
  myTraj.reserve(hits.size());

  TSOS predTsos(firstPredTsos);
  TSOS currTsos;

  int hitcounter = 1;
  for(RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end(); ++ihit, ++hitcounter) {

    const TransientTrackingRecHit & hit = (**ihit);

    // if unlikely(hit.det() == nullptr) continue;

    if unlikely( (!hit.isValid()) && hit.surface() == nullptr) {
       LogDebug("TrackFitters")<< " Error: invalid hit with no GeomDet attached .... skipping";
      continue;
    }
   //if (hit.det() && hit.geographicalId()<1000U) LogDebug("TrackFitters")<< "Problem 0 det id for " << typeid(hit).name() << ' ' <<  hit.det()->geographicalId() ;
   //if (hit.isValid() && hit.geographicalId()<1000U) LogDebug("TrackFitters")<< "Problem 0 det id for " << typeid(hit).name() << ' ' <<  hit.det()->geographicalId();

#ifdef EDM_ML_DEBUG
    if (hit.isValid()) {
      LogTrace("TrackFitters")<< " ----------------- HIT #" << hitcounter << " (VALID)-----------------------\n"
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

      LogTrace("TrackFitters") << " hit det=" << hitId.rawId();

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

    } else {
      LogTrace("TrackFitters")
	<< " ----------------- INVALID HIT #" << hitcounter << " -----------------------";
    }
#endif

    if ( hitcounter != 1) //no propagation needed for the first hit
      predTsos = p_cloned->propagate( currTsos, *(hit.surface()) );


    if unlikely(!predTsos.isValid()) {
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
	return Trajectory();
      }
    }


    if likely(hit.isValid()) {
        assert( (hit.geographicalId()!=0U) | !hit.canImproveWithTrack() ) ;
       	assert(hit.surface()!=nullptr);
	//update
	LogTrace("TrackFitters") << "THE HIT IS VALID: updating hit with predTsos";
        assert( (!(*ihit)->canImproveWithTrack()) | (nullptr!=theHitCloner));
        assert( (!(*ihit)->canImproveWithTrack()) | (nullptr!=dynamic_cast<BaseTrackerRecHit const*>((*ihit).get())));
	auto preciseHit = theHitCloner->makeShared(*ihit,predTsos);
        assert(preciseHit->isValid());
       	assert( (preciseHit->geographicalId()!=0U)  | (!preciseHit->canImproveWithTrack()) );
       	assert(preciseHit->surface()!=nullptr);

	if unlikely(!preciseHit->isValid()){
	    LogTrace("TrackFitters") << "THE Precise HIT IS NOT VALID: using currTsos = predTsos" << "\n";
	    currTsos = predTsos;
	    myTraj.push(TM(predTsos, *ihit,0,theGeometry->idToLayer((*ihit)->geographicalId()) ));

	  }else{
	  LogTrace("TrackFitters") << "THE Precise HIT IS VALID: updating currTsos" << "\n";
	  currTsos = updator()->update(predTsos, *preciseHit);
	  //check for valid hits with no det (refitter with constraints)
	  bool badState = (!currTsos.isValid())
          || (hit.geographicalId().det() == DetId::Tracker
              &&
              (std::abs(currTsos.localParameters().qbp())>100
               || std::abs(currTsos.localParameters().position().y()) > 1000
               || std::abs(currTsos.localParameters().position().x()) > 1000
               ) ) || edm::isNotFinite(currTsos.localParameters().qbp());
	  if unlikely(badState){
	    if (!currTsos.isValid()) {
	      edm::LogError("FailedUpdate") <<"updating with the hit failed. Not updating the trajectory with the hit";

            } 
	    else if (edm::isNotFinite(currTsos.localParameters().qbp())) {
              edm::LogError("TrajectoryNaN")<<"Trajectory has NaN";

            }
	    else{ 
              LogTrace("FailedUpdate")<<"updated state is valid but pretty bad, skipping. currTsos " <<currTsos<<"\n predTsos "<<predTsos;
            }
	    myTraj.push(TM(predTsos, *ihit,0,theGeometry->idToLayer((*ihit)->geographicalId())  ));
	    //There is a no-fail policy here. So, it's time to give up
	    //Keep the traj with invalid TSOS so that it's clear what happened
	    if( myTraj.foundHits() >= minHits_ ) {
	      LogDebug("TrackFitters") << " breaking trajectory" << "\n";
	      break;
	    } else {
	      LogDebug("TrackFitters") << " killing trajectory" << "\n";
	      return Trajectory();
	    }
	  } else{
	    if (preciseHit->det()){
	      myTraj.push(TM(predTsos, currTsos, preciseHit,
						  estimator()->estimate(predTsos, *preciseHit).second,
						  theGeometry->idToLayer(preciseHit->geographicalId())  ));
            }
	    else{
               myTraj.push(TM(predTsos, currTsos, preciseHit,
				estimator()->estimate(predTsos, *preciseHit).second));
            }
	  }
	}
      } else {
      //no update
      LogDebug("TrackFitters") << "THE HIT IS NOT VALID: using currTsos" << "\n";
      currTsos = predTsos;
      assert( ((*ihit)->det()==nullptr) || (*ihit)->geographicalId()!=0U);
      if ((*ihit)->det()) myTraj.push(TM(predTsos, *ihit,0,theGeometry->idToLayer((*ihit)->geographicalId())  ));
      else   myTraj.push(TM(predTsos, *ihit,0));
    }
    LogTrace("TrackFitters")
      << "predTsos !" << "\n"
      << predTsos 
      <<" with local position " << predTsos.localPosition()
      <<"currTsos !" << "\n"
      << currTsos
      <<" with local position " << currTsos.localPosition();
  }

  LogDebug("TrackFitters") << "Found 1 trajectory with " << myTraj.foundHits() << " valid hits\n";

  return ret;
}

