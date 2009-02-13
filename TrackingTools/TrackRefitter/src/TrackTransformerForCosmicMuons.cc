#include "TrackingTools/TrackRefitter/interface/TrackTransformerForCosmicMuons.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"


#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"


using namespace std;
using namespace edm;

/// Constructor
TrackTransformerForCosmicMuons::TrackTransformerForCosmicMuons(const ParameterSet& parameterSet){
  
  theTrackerRecHitBuilderName = parameterSet.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = parameterSet.getParameter<string>("MuonRecHitBuilder");

  theRPCInTheFit = parameterSet.getParameter<bool>("RefitRPCHits");

  theCacheId_TC = theCacheId_GTG = theCacheId_MG = theCacheId_TRH = 0;
}

/// Destructor
TrackTransformerForCosmicMuons::~TrackTransformerForCosmicMuons(){}


void TrackTransformerForCosmicMuons::setServices(const EventSetup& setup){
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  unsigned long long newCacheId_TC = setup.get<TrackingComponentsRecord>().cacheIdentifier();

  if ( newCacheId_TC != theCacheId_TC ){
    LogTrace(metname) << "Tracking Component changed!";
    theCacheId_TC = newCacheId_TC;
    
    setup.get<TrackingComponentsRecord>().get("KFFitterForRefitInsideOut",theFitterIO);
    setup.get<TrackingComponentsRecord>().get("KFSmootherForRefitInsideOut",theSmootherIO);
    setup.get<TrackingComponentsRecord>().get("SmartPropagatorRK",thePropagatorIO);

    setup.get<TrackingComponentsRecord>().get("KFFitterForRefitOutsideIn",theFitterOI);
    setup.get<TrackingComponentsRecord>().get("KFSmootherForRefitOutsideIn",theSmootherOI);
    setup.get<TrackingComponentsRecord>().get("SmartPropagatorRKOpposite",thePropagatorOI);

  }

  // Global Tracking Geometry
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if ( newCacheId_GTG != theCacheId_GTG ) {
    LogTrace(metname) << "GlobalTrackingGeometry changed!";
    theCacheId_GTG = newCacheId_GTG;
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  }
  
  // Magfield Field
  unsigned long long newCacheId_MG = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if ( newCacheId_MG != theCacheId_MG ) {
    LogTrace(metname) << "Magnetic Field changed!";
    theCacheId_MG = newCacheId_MG;
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }
  
  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if ( newCacheId_TRH != theCacheId_TRH ) {
    theCacheId_TRH = newCacheId_TRH;
    LogTrace(metname) << "TransientRecHitRecord changed!";
    setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);
  }
}


TransientTrackingRecHit::ConstRecHitContainer
TrackTransformerForCosmicMuons::getTransientRecHits(const reco::TransientTrack& track) const {

  TransientTrackingRecHit::ConstRecHitContainer tkHits;
  TransientTrackingRecHit::ConstRecHitContainer staHits;
  TransientTrackingRecHit::ConstRecHitContainer staHitsDTp;
  TransientTrackingRecHit::ConstRecHitContainer staHitsDTm;
  TransientTrackingRecHit::ConstRecHitContainer staHitsCSC;

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Tracker )
		tkHits.push_back(theTrackerRecHitBuilder->build(&**hit));
      else if ( (*hit)->geographicalId().det() == DetId::Muon ){
		if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  	LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  	continue;
		}
    	DetId hitId = (*hit)->geographicalId();
		if ( hitId.subdetId() == MuonSubdetId::DT ) {  
		    DTWireId id(hitId.rawId());
  			TransientTrackingRecHit::ConstRecHitContainer temp0;
			temp0.push_back(theMuonRecHitBuilder->build(&**hit));
			float gpZ = temp0.back()->globalPosition().z();	
			if (gpZ >= 0) staHitsDTp.push_back(theMuonRecHitBuilder->build(&**hit));
			if (gpZ < 0 ) staHitsDTm.push_back(theMuonRecHitBuilder->build(&**hit));
		}
		if ( hitId.subdetId() == MuonSubdetId::CSC ) {  
  			staHitsCSC.push_back(theMuonRecHitBuilder->build(&**hit));
			
		}
//	staHits.push_back(theMuonRecHitBuilder->build(&**hit));
      }
  
  	stable_sort(staHitsDTp	.begin(),staHitsDTp	.end(), ZedComparatorInOut() );
//  	reverse(staHitsDTp	.begin(),staHitsDTp	.end());
  	stable_sort(staHitsDTp	.begin(),staHitsDTp	.end(), ZedComparatorInOut() );
 
 	stable_sort(staHitsDTm	.begin(),staHitsDTm	.end(), ZedComparatorInOut() );
  //	reverse(staHitsDTm	.begin(),staHitsDTm	.end());
  	stable_sort(staHitsDTm	.begin(),staHitsDTm	.end(), ZedComparatorInOut() );

	if ( staHitsDTp.empty() ) {

		copy(staHitsDTm.begin(), staHitsDTm.end(), back_inserter(staHits));
	}
	else if ( staHitsDTm.empty() ) {

		copy(staHitsDTp.begin(), staHitsDTp.end(), back_inserter(staHits));
	}

	else if ( fabs(staHitsDTp.front()->globalPosition().z()) > fabs(staHitsDTm.front()->globalPosition().z())) {

		copy(staHitsDTm.end(), staHitsDTm.begin(), back_inserter(staHits));
		copy(staHitsDTp.begin(), staHitsDTp.end(), back_inserter(staHits));
	} else {

		copy(staHitsDTp.end(), staHitsDTp.begin(), back_inserter(staHits));
		copy(staHitsDTm.begin(), staHitsDTm.end(), back_inserter(staHits));
		
	}
  if(staHits.empty()) return staHits;

//
// put the DTs and CSC together
//

  copy(staHits.begin(),staHits.end(),back_inserter(tkHits));

  stable_sort(staHitsCSC.begin(), staHitsCSC.end(), ZedComparatorInOut());
  copy(staHitsCSC.begin(), staHitsCSC.end(), back_inserter(tkHits));

//	std::cout<<"dumping the rec Hits"<<std::endl;
  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator hit = tkHits.begin();
      hit !=tkHits.end(); ++hit){

    DetId hitId = (*hit)->geographicalId();
    GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();

    if(hitId.det() == DetId::Tracker) {
      if (hitId.subdetId() == StripSubdetector::TIB )  
	LogTrace("TrackFitters") << glbpoint << " I am TIB " << TIBDetId(hitId).layer();
      else if (hitId.subdetId() == StripSubdetector::TOB ) 
	LogTrace("TrackFitters") << glbpoint << " I am TOB " << TOBDetId(hitId).layer();
      else if (hitId.subdetId() == StripSubdetector::TEC ) 
	LogTrace("TrackFitters") << glbpoint << " I am TEC " << TECDetId(hitId).wheel();
      else if (hitId.subdetId() == StripSubdetector::TID ) 
	LogTrace("TrackFitters") << glbpoint << " I am TID " << TIDDetId(hitId).wheel();
      else if (hitId.subdetId() == StripSubdetector::TID ) 
	LogTrace("TrackFitters") << glbpoint << " I am TID " << TIDDetId(hitId).wheel();
      else if (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel ) 
	LogTrace("TrackFitters") << glbpoint << " I am PixBar " << PXBDetId(hitId).layer();
      else if (hitId.subdetId() == (int) PixelSubdetector::PixelEndcap )
	LogTrace("TrackFitters") << glbpoint << " I am PixFwd " << PXFDetId(hitId).disk();
      else 
	LogTrace("TrackFitters") << " UNKNOWN TRACKER HIT TYPE ";
    } else if(hitId.det() == DetId::Muon) {
      if(hitId.subdetId() == MuonSubdetId::DT)
	LogTrace("TrackFitters") << glbpoint << " I am DT " << DTWireId(hitId);
//	std::cout << glbpoint << " I am DT " << DTWireId(hitId)<<std::endl;
      else if (hitId.subdetId() == MuonSubdetId::CSC )
	LogTrace("TrackFitters") << glbpoint << " I am CSC " << CSCDetId(hitId);
//	std::cout<< glbpoint << " I am CSC " << CSCDetId(hitId)<<std::endl;
      else if (hitId.subdetId() == MuonSubdetId::RPC )
	LogTrace("TrackFitters") << glbpoint << " I am RPC " << RPCDetId(hitId);
      else 
	LogTrace("TrackFitters") << " UNKNOWN MUON HIT TYPE ";
    } else
      LogTrace("TrackFitters") << " UNKNOWN HIT TYPE ";
  } 
  
  return tkHits;
}


/// the refitter used to refit the reco::Track
ESHandle<TrajectoryFitter> TrackTransformerForCosmicMuons::fitter(bool up, float slope, float zFirst, bool cross) const{
//  if(up) return theFitterOI;
//  else return theFitterIO;
  	if (up){
		if (slope < 0 && zFirst > 0) 			return theFitterIO;
		if (slope < 0 && zFirst < 0 && cross) 	return theFitterIO;
		if (slope > 0 && zFirst < 0) 			return theFitterIO;
		if (slope > 0 && zFirst > 0 && cross) 	return theFitterIO;
  		else return theFitterOI;

	}else {
		if (slope > 0 && zFirst > 0) 			return theFitterOI;
		if (slope > 0 && zFirst < 0 && cross) 	return theFitterOI;
		if (slope < 0 && zFirst < 0) 			return theFitterOI;
		if (slope < 0 && zFirst > 0 && cross) 	return theFitterOI;
		else return theFitterIO;
	}
}
  
/// the smoother used to smooth the trajectory which came from the refitting step
ESHandle<TrajectorySmoother> TrackTransformerForCosmicMuons::smoother(bool up, float slope, float zFirst, bool cross) const{
//  if(up) return theSmootherOI;
//  else return theSmootherIO;
  	if (up){
		if (slope < 0 && zFirst > 0) 			return theSmootherIO;
		if (slope < 0 && zFirst < 0 && cross) 	return theSmootherIO;
		if (slope > 0 && zFirst < 0) 			return theSmootherIO;
		if (slope > 0 && zFirst > 0 && cross) 	return theSmootherIO;
  		else return theSmootherOI;

	}else {
		if (slope > 0 && zFirst > 0) 			return theSmootherOI;
		if (slope > 0 && zFirst < 0 && cross) 	return theSmootherOI;
		if (slope < 0 && zFirst < 0) 			return theSmootherOI;
		if (slope < 0 && zFirst > 0 && cross) 	return theSmootherOI;
		else return theSmootherIO;
	}
}

ESHandle<Propagator> TrackTransformerForCosmicMuons::propagator(bool up, float slope, float zFirst, bool cross) const{
//  if(up) return thePropagatorIO;
//  else return thePropagatorOI;
  	if (up){
		if (slope < 0 && zFirst > 0) 			return thePropagatorIO;
		if (slope < 0 && zFirst < 0 && cross) 	return thePropagatorIO;
		if (slope > 0 && zFirst < 0) 			return thePropagatorIO;
		if (slope > 0 && zFirst > 0 && cross) 	return thePropagatorIO;
  		else return thePropagatorOI;

	}else {
		if (slope > 0 && zFirst > 0) 			return thePropagatorOI;
		if (slope > 0 && zFirst < 0 && cross) 	return thePropagatorOI;
		if (slope < 0 && zFirst < 0) 			return thePropagatorOI;
		if (slope < 0 && zFirst > 0 && cross) 	return thePropagatorOI;
		else return thePropagatorIO;
	}
}



/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformerForCosmicMuons::transform(const reco::Track& tr) const {

  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  reco::TransientTrack track(tr,magneticField(),trackingGeometry());   

  // Build the transient Rechits
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit = getTransientRecHits(track);

  if(recHitsForReFit.size() < 2) return vector<Trajectory>();

  bool up = recHitsForReFit.back()->globalPosition().y()>0 ? true : false;
  LogTrace(metname) << "Up ? " << up;

  GlobalPoint gpfirst = recHitsForReFit.front()->globalPosition();
  GlobalPoint gplast = recHitsForReFit.back()->globalPosition();

	float slope = (gpfirst.y() - gplast.y())/(gpfirst.z() - gplast.z());
	float zfirst= gpfirst.z();
	bool cross = (gpfirst.z() * gplast.z()) > 0? false:true;

  LogTrace(metname)<<"slope = "<<slope;//<<std::endl;
  LogTrace(metname)<<"z first = "<<zfirst;//<<std::endl;

//  bool getProp;

  PropagationDirection propagationDirection = up ? oppositeToMomentum : alongMomentum;
  if ( up 	&& slope < 0 && zfirst > 0) 			propagationDirection = alongMomentum;
  if ( up 	&& slope < 0 && zfirst < 0 && cross) 	propagationDirection = alongMomentum;
  if ( up 	&& slope > 0 && zfirst < 0) 			propagationDirection = alongMomentum;
  if ( up 	&& slope > 0 && zfirst > 0 && cross) 	propagationDirection = alongMomentum;
  if ( !up 	&& slope > 0 && zfirst > 0) 			propagationDirection = oppositeToMomentum;
  if ( !up 	&& slope > 0 && zfirst < 0 && cross) 	propagationDirection = oppositeToMomentum;
  if ( !up 	&& slope < 0 && zfirst < 0) 			propagationDirection = oppositeToMomentum;
  if ( !up 	&& slope < 0 && zfirst > 0 && cross) 	propagationDirection = oppositeToMomentum;

  TrajectoryStateOnSurface firstTSOS = up ? track.outermostMeasurementState() : track.innermostMeasurementState();
  if (up && slope < 0 && zfirst > 0) 			firstTSOS = track.innermostMeasurementState();
  if (up && slope < 0 && zfirst < 0 && cross) 	firstTSOS = track.innermostMeasurementState();
  if (up && slope > 0 && zfirst < 0) 			firstTSOS = track.innermostMeasurementState();
  if (up && slope > 0 && zfirst > 0 && cross) 	firstTSOS = track.innermostMeasurementState();
  if (!up && slope > 0 && zfirst > 0) 			firstTSOS = track.outermostMeasurementState();
  if (!up && slope > 0 && zfirst < 0 && cross)	firstTSOS = track.outermostMeasurementState();
  if (!up && slope < 0 && zfirst < 0) 			firstTSOS = track.outermostMeasurementState();
  if (!up && slope < 0 && zfirst > 0 && cross) 	firstTSOS = track.outermostMeasurementState();

  unsigned int innerId = up ? track.track().outerDetId() : track.track().innerDetId();
  if (up && slope < 0 && zfirst > 0) 			innerId = track.track().innerDetId();
  if (up && slope < 0 && zfirst < 0 && cross) 	innerId = track.track().innerDetId();
  if (up && slope < 0 && zfirst > 0) 			innerId = track.track().innerDetId();
  if (up && slope < 0 && zfirst < 0 && cross) 	innerId = track.track().innerDetId();
  if (!up && slope > 0 && zfirst > 0) 			innerId = track.track().outerDetId();
  if (!up && slope > 0 && zfirst < 0 && cross) 	innerId = track.track().outerDetId();
  if (!up && slope > 0 && zfirst > 0) 			innerId = track.track().outerDetId();
  if (!up && slope > 0 && zfirst < 0 && cross) 	innerId = track.track().outerDetId();

//
//
//
//	unsigned int innerId = recHitsForReFit.front()->geographicalId();

  LogTrace(metname) << "Prop Dir: " << propagationDirection << " FirstId " << innerId << " firstTSOS " << firstTSOS;

  TrajectorySeed seed(PTrajectoryStateOnDet(),TrajectorySeed::recHitContainer(),propagationDirection);


  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogTrace(metname)<<"Propagation occurring"<<endl;
    firstTSOS = propagator(up, slope, zfirst, cross)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    LogTrace(metname)<<"Final destination: " << recHitsForReFit.front()->det()->surface().position() << endl;
    if(!firstTSOS.isValid()){
      LogTrace(metname)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }
  

  vector<Trajectory> trajectories = fitter(up, slope, zfirst, cross)->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogTrace(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory trajectoryBW = trajectories.front();
    
  vector<Trajectory> trajectoriesSM = smoother(up, slope, zfirst, cross)->trajectories(trajectoryBW);

  if(trajectoriesSM.empty()){
    LogTrace(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
  }
  
  return trajectoriesSM;

}


