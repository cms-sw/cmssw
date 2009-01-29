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

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Tracker )continue;
//	tkHits.push_back(theTrackerRecHitBuilder->build(&**hit));
      else if ( (*hit)->geographicalId().det() == DetId::Muon ){
		if( (*hit)->geographicalId().subdetId() == MuonSubdetId::RPC && !theRPCInTheFit){
	  	LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  	continue;
		}
		staHits.push_back(theMuonRecHitBuilder->build(&**hit));
      }
  
  if(staHits.empty()) return staHits;

  	GlobalPoint pfirst = staHits.front()->globalPosition();
  	GlobalPoint plast  = staHits.back()->globalPosition();


//  bool up = staHits.front()->globalPosition().y()>0 ? true : false;
  	bool up = pfirst.y()>0 ? true : false;

  	float slope = (plast.y() - pfirst.y()) / (plast.z() - pfirst.z());

  	bool ZedResort = false;
	
	if ( fabs(plast.z()) < fabs(pfirst.z())) {
		 ZedResort = true;
		 LogTrace("TrackFitters")<<"RESORTING THE HITS IN Z, DT CSC RPC"<<std::endl;
	}

//  if(up){
	if(ZedResort){
    	reverse(staHits.begin(),staHits.end());
    	reverse(tkHits.begin(),tkHits.end());
  	}

  copy(staHits.begin(),staHits.end(),back_inserter(tkHits));

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
ESHandle<TrajectoryFitter> TrackTransformerForCosmicMuons::fitter(bool up) const{
  if(up) return theFitterOI;
  else return theFitterIO;
}
  
/// the smoother used to smooth the trajectory which came from the refitting step
ESHandle<TrajectorySmoother> TrackTransformerForCosmicMuons::smoother(bool up) const{
  if(up) return theSmootherOI;
  else return theSmootherIO;
}

ESHandle<Propagator> TrackTransformerForCosmicMuons::propagator(bool up) const{
  if(up) return thePropagatorIO;
  else return thePropagatorOI;
}



/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformerForCosmicMuons::transform(const reco::Track& tr) const {

  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  reco::TransientTrack track(tr,magneticField(),trackingGeometry());   

  // Build the transient Rechits
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit = getTransientRecHits(track);

  if(recHitsForReFit.size() < 2) return vector<Trajectory>();

//  bool up = recHitsForReFit.back()->globalPosition().y()>0 ? true : false;
//  LogTrace(metname) << "Up ? " << up;


  	GlobalPoint pfirst = recHitsForReFit.front()->globalPosition();
  	GlobalPoint plast  = recHitsForReFit.back()->globalPosition();
 
 	bool up = plast.y()>0 ? true : false;
  	LogTrace(metname) << "Up ? " << up;
	bool leftRight = (plast.z() > pfirst.z())? true:false;

  	float slope = (plast.y() - pfirst.y());
//	if ( plast.z() - pfirst.z() != 0) slope = slope / (plast.z() - pfirst.z()); 

  	bool ZedResort = false;
	
	if ( fabs(plast.z()) < fabs(pfirst.z())) {
		 ZedResort = true;
		 LogTrace("TrackFitters")<<"RESORTING THE HITS IN Z, Failure"<<std::endl;
	}

  	LogTrace(metname) << "slope Failure? " << slope;
  	LogTrace(metname) << "leftRight ? Failure" << leftRight;


  	PropagationDirection propagationDirection = up ? oppositeToMomentum : alongMomentum;
  	TrajectoryStateOnSurface firstTSOS = up ? track.outermostMeasurementState() : track.innermostMeasurementState();
  	unsigned int innerId = up ? track.track().outerDetId() : track.track().innerDetId();


	if ( ZedResort ) {

				propagationDirection = alongMomentum;
				firstTSOS = track.innermostMeasurementState();
				innerId = track.track().innerDetId();

	}


  LogTrace(metname) << "Prop Dir: " << propagationDirection << " FirstId " << innerId << " firstTSOS " << firstTSOS;

  TrajectorySeed seed(PTrajectoryStateOnDet(),TrajectorySeed::recHitContainer(),propagationDirection);


  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogTrace(metname)<<"Propagation occurring"<<endl;
    firstTSOS = propagator(up && !ZedResort)->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    LogTrace(metname)<<"Final destination: " << recHitsForReFit.front()->det()->surface().position() << endl;
    if(!firstTSOS.isValid()){
      LogTrace(metname)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }
  

  vector<Trajectory> trajectories = fitter(up && !ZedResort)->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogTrace(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory trajectoryBW = trajectories.front();
    
  vector<Trajectory> trajectoriesSM = smoother(up && !ZedResort)->trajectories(trajectoryBW);

  if(trajectoriesSM.empty()){
    LogTrace(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
  }
  
  return trajectoriesSM;

}


