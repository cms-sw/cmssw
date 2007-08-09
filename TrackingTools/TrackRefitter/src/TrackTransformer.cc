#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

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

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

using namespace std;
using namespace edm;

/// Constructor
TrackTransformer::TrackTransformer(const ParameterSet& parameterSet){
  
  // Refit direction
  string refitDirectionName = parameterSet.getParameter<string>("RefitDirection");
  
  if (refitDirectionName == "insideOut" ) theRefitDirection = insideOut;
    else if (refitDirectionName == "outsideIn" ) theRefitDirection = outsideIn;
    else 
      throw cms::Exception("TrackTransformer constructor") 
	<<"Wrong refit direction chosen in TrackTransformer ParameterSet"
	<< "\n"
	<< "Possible choices are:"
	<< "\n"
	<< "RefitDirection = insideOut or RefitDirection = outsideIn";
  
  theFitterName = parameterSet.getParameter<string>("Fitter");  
  theSmootherName = parameterSet.getParameter<string>("Smoother");  
  thePropagatorName = parameterSet.getParameter<string>("Propagator");

  theTrackerRecHitBuilderName = parameterSet.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = parameterSet.getParameter<string>("MuonRecHitBuilder");

  theRPCInTheFit = parameterSet.getParameter<bool>("RefitRPCHits");

  theCacheId_TC = theCacheId_GTG = theCacheId_MG = theCacheId_TRH = 0;
}

/// Destructor
TrackTransformer::~TrackTransformer(){}


void TrackTransformer::setServices(const EventSetup& setup){
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  unsigned long long newCacheId_TC = setup.get<TrackingComponentsRecord>().cacheIdentifier();

  if ( newCacheId_TC != theCacheId_TC ){
    LogDebug(metname) << "Tracking Component changed!";
    theCacheId_TC = newCacheId_TC;
    
    setup.get<TrackingComponentsRecord>().get(theFitterName,theFitter);
    setup.get<TrackingComponentsRecord>().get(theSmootherName,theSmoother);

    setup.get<TrackingComponentsRecord>().get(thePropagatorName,thePropagator);
  }

  // Global Tracking Geometry
  unsigned long long newCacheId_GTG = setup.get<GlobalTrackingGeometryRecord>().cacheIdentifier();
  if ( newCacheId_GTG != theCacheId_GTG ) {
    LogDebug(metname) << "GlobalTrackingGeometry changed!";
    theCacheId_GTG = newCacheId_GTG;
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
  }
  
  // Magfield Field
  unsigned long long newCacheId_MG = setup.get<IdealMagneticFieldRecord>().cacheIdentifier();
  if ( newCacheId_MG != theCacheId_MG ) {
    LogDebug(metname) << "Magnetic Field changed!";
    theCacheId_MG = newCacheId_MG;
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }
  
  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if ( newCacheId_TRH != theCacheId_TRH ) {
    LogDebug(metname) << "TransientRecHitRecord changed!";
    setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName,theTrackerRecHitBuilder);
    setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName,theMuonRecHitBuilder);
  }
}


vector<Trajectory> TrackTransformer::transform(const reco::TrackRef& track) const {
  return transform(*track);
}


TransientTrackingRecHit::ConstRecHitContainer
TrackTransformer::getTransientRecHits(const reco::TransientTrack& track) const {

  TransientTrackingRecHit::ConstRecHitContainer result;
  
  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Tracker )
	result.push_back(theTrackerRecHitBuilder->build(&**hit));
      else if ( (*hit)->geographicalId().det() == DetId::Muon ){
	if( (*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit){
	  LogDebug("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged"; 
	  continue;
	}
	result.push_back(theMuonRecHitBuilder->build(&**hit));
      }
  
  return result;
}

// FIXME: check this method!
TrackTransformer::RefitDirection 
TrackTransformer::checkRecHitsOrdering(TransientTrackingRecHit::ConstRecHitContainer& recHits) const {
 
 if (!recHits.empty()){
    double rFirst = recHits.front()->globalPosition().mag();
    double rLast  = recHits.back()->globalPosition().mag();
    if(rFirst < rLast) return insideOut;
    else if(rFirst > rLast) return outsideIn;
    else{
      LogError("Reco|TrackingTools|TrackTransformer") << "Impossible determine the rechits order" <<endl;
      return undetermined;
    }
  }
  else{
    LogError("Reco|TrackingTools|TrackTransformer") << "Impossible determine the rechits order" <<endl;
    return undetermined;
    }
}

/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformer::transform(const reco::Track& newTrack) const {
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  reco::TransientTrack track(newTrack,magneticField(),trackingGeometry());   
  
  // Build the transient Rechits
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit = getTransientRecHits(track);
  if(recHitsForReFit.size() < 2) return vector<Trajectory>();
  
  // Check the order of the rechits
  RefitDirection recHitsOrder = checkRecHitsOrdering(recHitsForReFit);

  // Reverse the order in the case of inconsistency between the fit direction and the rechit order
  if(theRefitDirection != recHitsOrder) reverse(recHitsForReFit.begin(),recHitsForReFit.end());

  // Fill the starting state
  TrajectoryStateOnSurface firstTSOS;
  unsigned int innerId;
  if(theRefitDirection == insideOut){
    innerId =   newTrack.innerDetId();
    firstTSOS = track.innermostMeasurementState();
  }
  else{
    innerId   = newTrack.outerDetId();
    firstTSOS = track.outermostMeasurementState();
  }

  if(!firstTSOS.isValid()){
    LogWarning(metname)<<"Error wrong initial state!"<<endl;
    return vector<Trajectory>();
  }

  // This is the only way to get a TrajectorySeed with settable propagation direction
  PTrajectoryStateOnDet garbage1;
  edm::OwnVector<TrackingRecHit> garbage2;
  PropagationDirection propDir = 
    (firstTSOS.globalPosition().basicVector().dot(firstTSOS.globalMomentum().basicVector())>0) ? alongMomentum : oppositeToMomentum;

  //  if(propDir == alongMomentum && theRefitDirection == insideOut)  OK;
  if(propDir == alongMomentum && theRefitDirection == outsideIn)  propDir=oppositeToMomentum;
  if(propDir == oppositeToMomentum && theRefitDirection == insideOut) propDir=alongMomentum;
  // if(propDir == oppositeToMomentum && theRefitDirection == outsideIn) OK;
  
  TrajectorySeed seed(garbage1,garbage2,propDir);

  if(recHitsForReFit.front()->geographicalId() != DetId(innerId)){
    LogDebug(metname)<<"Propagation occured"<<endl;
    firstTSOS = propagator()->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    if(!firstTSOS.isValid()){
      LogDebug(metname)<<"Propagation error!"<<endl;
      return vector<Trajectory>();
    }
  }

  vector<Trajectory> trajectories = theFitter->fit(seed,recHitsForReFit,firstTSOS);
  
  if(trajectories.empty()){
    LogDebug(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory trajectoryBW = trajectories.front();
    
  vector<Trajectory> trajectoriesSM = theSmoother->trajectories(trajectoryBW);
  
  if(trajectoriesSM.empty()){
    LogDebug(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
  }
  
  return trajectoriesSM;
}


