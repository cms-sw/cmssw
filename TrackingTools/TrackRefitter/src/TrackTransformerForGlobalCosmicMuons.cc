#include "TrackingTools/TrackRefitter/interface/TrackTransformerForGlobalCosmicMuons.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
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

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

using namespace std;
using namespace edm;

/// Constructor
TrackTransformerForGlobalCosmicMuons::TrackTransformerForGlobalCosmicMuons(const ParameterSet& parameterSet){
  
  theTrackerRecHitBuilderName = parameterSet.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = parameterSet.getParameter<string>("MuonRecHitBuilder");

  theRPCInTheFit = parameterSet.getParameter<bool>("RefitRPCHits");

  theCacheId_TC = theCacheId_GTG = theCacheId_MG = theCacheId_TRH = 0;
  theSkipStationDT      = parameterSet.getParameter<int>("SkipStationDT");
  theSkipStationCSC     = parameterSet.getParameter<int>("SkipStationCSC");
  theSkipWheelDT		= parameterSet.getParameter<int>("SkipWheelDT");
  theTrackerSkipSystem	= parameterSet.getParameter<int>("TrackerSkipSystem");
  theTrackerSkipSection	= parameterSet.getParameter<int>("TrackerSkipSection");//layer, wheel, or disk depending on the system 
}

/// Destructor
TrackTransformerForGlobalCosmicMuons::~TrackTransformerForGlobalCosmicMuons(){}


void TrackTransformerForGlobalCosmicMuons::setServices(const EventSetup& setup){
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";

  setup.get<TrajectoryFitter::Record>().get("KFFitterForRefitInsideOut",theFitterIO);
  setup.get<TrajectoryFitter::Record>().get("KFSmootherForRefitInsideOut",theSmootherIO);
  setup.get<TrajectoryFitter::Record>().get("KFFitterForRefitOutsideIn",theFitterOI);
  setup.get<TrajectoryFitter::Record>().get("KFSmootherForRefitOutsideIn",theSmootherOI);
  
  unsigned long long newCacheId_TC = setup.get<TrackingComponentsRecord>().cacheIdentifier();

  if ( newCacheId_TC != theCacheId_TC ){
    LogTrace(metname) << "Tracking Component changed!";
    theCacheId_TC = newCacheId_TC;
    setup.get<TrackingComponentsRecord>().get("SmartPropagatorRK",thePropagatorIO);
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

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup.get<IdealGeometryRecord>().get(tTopoHand);
  tTopo_=tTopoHand.product();
}


TransientTrackingRecHit::ConstRecHitContainer
TrackTransformerForGlobalCosmicMuons::getTransientRecHits(const reco::TransientTrack& track) const {

  TransientTrackingRecHit::ConstRecHitContainer tkHits;
  TransientTrackingRecHit::ConstRecHitContainer staHits;

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if((*hit)->isValid()) {
      if ( (*hit)->geographicalId().det() == DetId::Tracker && TrackerKeep((*hit)->geographicalId())) {
		tkHits.push_back(theTrackerRecHitBuilder->build(&**hit));
      } else if ( (*hit)->geographicalId().det() == DetId::Muon && MuonKeep((*hit)->geographicalId())){
		if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  	LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  	continue;
		}
		staHits.push_back(theMuonRecHitBuilder->build(&**hit));
      }
    }
  }
  
  if(staHits.empty()) return staHits;

/*
  bool up = staHits.front()->globalPosition().y()>0 ? true : false;

  if(up){
    reverse(staHits.begin(),staHits.end());
    reverse(tkHits.begin(),tkHits.end());
  }
*/
  copy(staHits.begin(),staHits.end(),back_inserter(tkHits));

  for(TransientTrackingRecHit::ConstRecHitContainer::const_iterator hit = tkHits.begin();
      hit !=tkHits.end(); ++hit){

    DetId hitId = (*hit)->geographicalId();
    GlobalPoint glbpoint = trackingGeometry()->idToDet(hitId)->position();

    if(hitId.det() == DetId::Tracker) {
      if (hitId.subdetId() == StripSubdetector::TIB )  
	LogTrace("TrackFitters") << glbpoint << " I am TIB " << tTopo_->tibLayer(hitId);
      else if (hitId.subdetId() == StripSubdetector::TOB ) 
	LogTrace("TrackFitters") << glbpoint << " I am TOB " << tTopo_->tobLayer(hitId);
      else if (hitId.subdetId() == StripSubdetector::TEC ) 
	LogTrace("TrackFitters") << glbpoint << " I am TEC " << tTopo_->tecWheel(hitId);
      else if (hitId.subdetId() == StripSubdetector::TID ) 
	LogTrace("TrackFitters") << glbpoint << " I am TID " << tTopo_->tidWheel(hitId);
      else if (hitId.subdetId() == (int) PixelSubdetector::PixelBarrel ) 
	LogTrace("TrackFitters") << glbpoint << " I am PixBar " << tTopo_->pxbLayer(hitId);
      else if (hitId.subdetId() == (int) PixelSubdetector::PixelEndcap )
	LogTrace("TrackFitters") << glbpoint << " I am PixFwd " << tTopo_->pxfDisk(hitId);
      else 
	LogTrace("TrackFitters") << " UNKNOWN TRACKER HIT TYPE ";
    } else if(hitId.det() == DetId::Muon) {
      if(hitId.subdetId() == MuonSubdetId::DT)
	LogTrace("TrackFitters") << glbpoint << " I am DT " << DTWireId(hitId);
      else if (hitId.subdetId() == MuonSubdetId::CSC )
	LogTrace("TrackFitters") << glbpoint << " I am CSC " << CSCDetId(hitId);
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
ESHandle<TrajectoryFitter> TrackTransformerForGlobalCosmicMuons::fitter(bool up) const{
  if(up) return theFitterOI;
  else return theFitterIO;
}
  
/// the smoother used to smooth the trajectory which came from the refitting step
ESHandle<TrajectorySmoother> TrackTransformerForGlobalCosmicMuons::smoother(bool up) const{
  if(up) return theSmootherOI;
  else return theSmootherIO;
}

ESHandle<Propagator> TrackTransformerForGlobalCosmicMuons::propagator(bool up) const{
  if(up) return thePropagatorIO;
  else return thePropagatorOI;
}



/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformerForGlobalCosmicMuons::transform(const reco::Track& tr) const {

  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  reco::TransientTrack track(tr,magneticField(),trackingGeometry());   

  // Build the transient Rechits
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit = getTransientRecHits(track);

  if(recHitsForReFit.size() < 2) return vector<Trajectory>();

  bool up = recHitsForReFit.back()->globalPosition().y()>0 ? true : false;
  LogTrace(metname) << "Up ? " << up;

  PropagationDirection propagationDirection = up ? oppositeToMomentum : alongMomentum;
  TrajectoryStateOnSurface firstTSOS = up ? track.outermostMeasurementState() : track.innermostMeasurementState();
  unsigned int innerId = up ? track.track().outerDetId() : track.track().innerDetId();

  LogTrace(metname) << "Prop Dir: " << propagationDirection << " FirstId " << innerId << " firstTSOS " << firstTSOS;

  TrajectorySeed seed(PTrajectoryStateOnDet(),TrajectorySeed::recHitContainer(),propagationDirection);


  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogTrace(metname)<<"Propagation occurring"<<endl;
    firstTSOS = propagator(up)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    LogTrace(metname)<<"Final destination: " << recHitsForReFit.front()->det()->surface().position() << endl;
    if(!firstTSOS.isValid()){
      LogTrace(metname)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }
  

  vector<Trajectory> trajectories = fitter(up)->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogTrace(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory trajectoryBW = trajectories.front();
    
  vector<Trajectory> trajectoriesSM = smoother(up)->trajectories(trajectoryBW);

  if(trajectoriesSM.empty()){
    LogTrace(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
  }
  
  return trajectoriesSM;

}

//
// Selection for Tracker Hits
//
bool TrackTransformerForGlobalCosmicMuons::TrackerKeep(DetId id) const{
	
  if (id.det() != DetId::Tracker ) return false;
  if (theTrackerSkipSystem < 0 ) return true;
  bool retVal = true;

  int layer = -999;
  
  if ( id.subdetId() == theTrackerSkipSystem)
    layer=tTopo_->layer(id);
  
  if (theTrackerSkipSection > -998 && layer == theTrackerSkipSection) retVal = false;
  
  return retVal;
}
//
// Selection for Muon hits
//
bool TrackTransformerForGlobalCosmicMuons::MuonKeep(DetId id) const {

	if (id.det() != DetId::Muon) return false;
	if (theSkipStationDT < 0 && theSkipStationCSC < 0) return true;

	int station = -999;
	int wheel	= -999;
	bool isRPC 	= false;
	bool isDT	= false;
	bool isCSC	= false;
		
    if ( id.subdetId() == MuonSubdetId::DT ) {
  		DTChamberId did(id.rawId());
  		station = did.station();
  		wheel = did.wheel();
		isDT = true;
    } else if ( id.subdetId() == MuonSubdetId::CSC ) {
  		CSCDetId did(id.rawId());
  		station = did.station();
		isCSC = true;
    } else if ( id.subdetId() == MuonSubdetId::RPC ) {
  		RPCDetId rpcid(id.rawId());
  		station = rpcid.station();
		isRPC = true;
    }
	
	if (isRPC 	&& (station	== theSkipStationCSC || station == theSkipStationDT)) return false;
	if (isDT 	&& station 	== theSkipStationDT		) return false;
	if (isCSC 	&& station 	== theSkipStationCSC	) return false;

	if (isDT && theSkipWheelDT > -998 && wheel == theSkipWheelDT) return false;
	
	return true;
}
