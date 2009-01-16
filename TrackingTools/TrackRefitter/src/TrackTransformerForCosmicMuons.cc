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
  theSkipStation        = parameterSet.getParameter<int>("SkipStation");
  theKeepDTWheel        = parameterSet.getParameter<int>("KeepDTWheel");
  theTrackerSkipSystem	= parameterSet.getParameter<int>("TrackerSkipSystem");
  theTrackerSkipSection	= parameterSet.getParameter<int>("TrackerSkipSection");//layer, wheel, or disk depending on the system
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

	for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
		if((*hit)->isValid())
      		if ( (*hit)->geographicalId().det() == DetId::Tracker ) {
				DetId id = (*hit)->geographicalId();
				if ( TrackerKeep(id)) tkHits.push_back(theTrackerRecHitBuilder->build(&**hit));
			}
    		else if ( (*hit)->geographicalId().det() == DetId::Muon ){
				if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  				LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  				continue;
				}
				DetId id = (*hit)->geographicalId();
				if ( MuonKeep(id)) staHits.push_back(theMuonRecHitBuilder->build(&**hit));
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

//  stable_sort(tkHits.begin(),tkHits.end(), ZedComparatorMinusPlus() );
  stable_sort(tkHits.begin(),tkHits.end(), ZedComparatorInOut() );

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
ESHandle<TrajectoryFitter> TrackTransformerForCosmicMuons::fitter(bool outIn) const{
  if(!outIn) return theFitterOI;
  else return theFitterIO;
}
  
/// the smoother used to smooth the trajectory which came from the refitting step
ESHandle<TrajectorySmoother> TrackTransformerForCosmicMuons::smoother(bool outIn) const{
  if(!outIn) return theSmootherOI;
  else return theSmootherIO;
}

ESHandle<Propagator> TrackTransformerForCosmicMuons::propagator(bool along) const{
  if(along) return thePropagatorIO;
  else return thePropagatorOI;
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

  float yleft = recHitsForReFit.front()->globalPosition().y();
  float yright= recHitsForReFit.back()->globalPosition().y();

  float zleft = recHitsForReFit.front()->globalPosition().z();
  float zright= recHitsForReFit.back()->globalPosition().z();

  float deltay = yleft - yright;
  float slopeVal = deltay/(zleft-zright);
//  bool slopePos = slopeVal>0?true:false;

  bool fitterSmootherIO = true;
  bool propcaseIO = true;

  if (slopeVal < 0 && yleft > 0) { fitterSmootherIO = false; 	propcaseIO = true;} 
  if (slopeVal < 0 && yleft < 0) { fitterSmootherIO = true; 	propcaseIO = true;} 
  if (slopeVal > 0 && yleft < 0) { fitterSmootherIO = false; 	propcaseIO = false;} 
  if (slopeVal > 0 && yleft > 0) { fitterSmootherIO = true; 	propcaseIO = false;} 

 // PropagationDirection propagationDirection = up ? oppositeToMomentum : alongMomentum;
  PropagationDirection propagationDirection = propcaseIO ? alongMomentum : oppositeToMomentum;
  TrajectoryStateOnSurface firstTSOS = up ? track.outermostMeasurementState() : track.innermostMeasurementState();
  unsigned int innerId = up ? track.track().outerDetId() : track.track().innerDetId();

  LogTrace(metname) << "Prop Dir: " << propagationDirection << " FirstId " << innerId << " firstTSOS " << firstTSOS;

  TrajectorySeed seed(PTrajectoryStateOnDet(),TrajectorySeed::recHitContainer(),propagationDirection);


  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogTrace(metname)<<"Propagation occurring"<<endl;
    firstTSOS = propagator(propcaseIO)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    LogTrace(metname)<<"Final destination: " << recHitsForReFit.front()->det()->surface().position() << endl;
    if(!firstTSOS.isValid()){
      LogTrace(metname)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }
  

  vector<Trajectory> trajectories = fitter(fitterSmootherIO)->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogTrace(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory trajectoryBW = trajectories.front();
    
  vector<Trajectory> trajectoriesSM = smoother(fitterSmootherIO)->trajectories(trajectoryBW);

  if(trajectoriesSM.empty()){
    LogTrace(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
  }
  
  return trajectoriesSM;

}

//
// Remove Selected Station Rec Hits
//
TransientTrackingRecHit::ConstRecHitContainer 
	TrackTransformerForCosmicMuons::getRidOfSelectStationHits(TransientTrackingRecHit::ConstRecHitContainer hits) const
{
  TransientTrackingRecHit::ConstRecHitContainer results;
  TransientTrackingRecHit::ConstRecHitContainer::const_iterator it = hits.begin();
  for (; it!=hits.end(); it++) {

    DetId id = (*it)->geographicalId();

    //Check that this is a Muon hit that we're toying with -- else pass on this because the hacker is a moron / not careful

    if (id.det() == DetId::Tracker && theTrackerSkipSystem > 0) {
      int layer = -999;
      int disk  = -999;
      int wheel = -999;
      if ( id.subdetId() == theTrackerSkipSystem){
	//                              continue;  //caveat that just removes the whole system from refitting

		if (theTrackerSkipSystem == PXB) {
	  		PXBDetId did(id.rawId());
	  		layer = did.layer();
		}
		if (theTrackerSkipSystem == TIB) {
	  		TIBDetId did(id.rawId());
	  		layer = did.layer();
		}	

		if (theTrackerSkipSystem == TOB) {
	  		TOBDetId did(id.rawId());
	  		layer = did.layer();
		}
		if (theTrackerSkipSystem == PXF) {
	  		PXFDetId did(id.rawId());
	  		disk = did.disk();
		}
		if (theTrackerSkipSystem == TID) {
	  		TIDDetId did(id.rawId());
	  		wheel = did.wheel();
		}
		if (theTrackerSkipSystem == TEC) {
	  		TECDetId did(id.rawId());
	  		wheel = did.wheel();
		}
		if (theTrackerSkipSection >= 0 && layer == theTrackerSkipSection) continue;
		if (theTrackerSkipSection >= 0 && disk == theTrackerSkipSection) continue;
		if (theTrackerSkipSection >= 0 && wheel == theTrackerSkipSection) continue;
      }
    }

    if (id.det() == DetId::Muon && theSkipStation) {
      int station = -999;
      int wheel = -999;
      if ( id.subdetId() == MuonSubdetId::DT ) {
		DTChamberId did(id.rawId());
		station = did.station();
		wheel = did.wheel();
      } else if ( id.subdetId() == MuonSubdetId::CSC ) {
		CSCDetId did(id.rawId());
		station = did.station();
      } else if ( id.subdetId() == MuonSubdetId::RPC ) {
		RPCDetId rpcid(id.rawId());
		station = rpcid.station();
      }
      if(station == theSkipStation) continue;
    }


    if ( id.det() == DetId::Tracker ) results.push_back(theTrackerRecHitBuilder	->build(&**it));
    if ( id.det() == DetId::Muon 	) results.push_back(theMuonRecHitBuilder	->build(&**it));
  }
  return results;
}


//
// Selection for Tracker Hits
//
bool TrackTransformerForCosmicMuons::TrackerKeep(DetId id) const{
	
	bool retVal = true;
	if (id.det() != DetId::Tracker ) return false;
	if (theTrackerSkipSystem < 0 ) return true;
	

    int layer = -999;
    int disk  = -999;
    int wheel = -999;


      if ( id.subdetId() == theTrackerSkipSystem){

		if (theTrackerSkipSystem == PXB) {
			PXBDetId did(id.rawId());
			layer = did.layer();
		}

		if (theTrackerSkipSystem == TIB) {
			TIBDetId did(id.rawId());
			layer = did.layer();
		}	

		if (theTrackerSkipSystem == TOB) {
			TOBDetId did(id.rawId());
			layer = did.layer();
		}
		if (theTrackerSkipSystem == PXF) {
			PXFDetId did(id.rawId());
			disk = did.disk();
		}
		if (theTrackerSkipSystem == TID) {
			TIDDetId did(id.rawId());
			wheel = did.wheel();
		}
		if (theTrackerSkipSystem == TEC) {
			TECDetId did(id.rawId());
			wheel = did.wheel();
		}
	}

	if (theTrackerSkipSection >= 0 && layer == theTrackerSkipSection) retVal = false;
	if (theTrackerSkipSection >= 0 && disk 	== theTrackerSkipSection) retVal = false;
	if (theTrackerSkipSection >= 0 && wheel == theTrackerSkipSection) retVal = false;


	return retVal;
}

//
// Selection for Muon hits
//

bool TrackTransformerForCosmicMuons::MuonKeep(DetId id) const {

	bool retVal = true;
	if (id.det() != DetId::Muon) return false;
	int station = -999;
	int wheel	= -999;
		
    if ( id.subdetId() == MuonSubdetId::DT ) {
  		DTChamberId did(id.rawId());
  		station = did.station();
  		wheel = did.wheel();
    } else if ( id.subdetId() == MuonSubdetId::CSC ) {
  		CSCDetId did(id.rawId());
  		station = did.station();
    } else if ( id.subdetId() == MuonSubdetId::RPC ) {
  		RPCDetId rpcid(id.rawId());
  		station = rpcid.station();
    }
	
	if ( station == theSkipStation) retVal = false;
	if ( theKeepDTWheel > -10)
		if ( wheel > -10)
			if ( fabs(wheel) != theKeepDTWheel) retVal = false;


	return retVal;
}


//DEFINE_ANOTHER_FWK_MODULE(TracksToTrajectoriesForCosmicMuons)
