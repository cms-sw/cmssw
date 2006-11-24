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
  
  theTrackerRecHitBuilderName = parameterSet.getParameter<string>("TrackerRecHitBuilder");
  theMuonRecHitBuilderName = parameterSet.getParameter<string>("MuonRecHitBuilder");

  theCacheId_TC = theCacheId_GTG = theCacheId_MG = theCacheId_TRH = 0;
}

/// Destructor
TrackTransformer::~TrackTransformer(){
}


void TrackTransformer::setServices(const EventSetup& setup){
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  unsigned long long newCacheId_TC = setup.get<TrackingComponentsRecord>().cacheIdentifier();

  if ( newCacheId_TC != theCacheId_TC ){
    LogDebug(metname) << "Tracking Component changed!";
    theCacheId_TC = newCacheId_TC;
    
    setup.get<TrackingComponentsRecord>().get(theFitterName,theFitter);
    setup.get<TrackingComponentsRecord>().get(theSmootherName,theSmoother);

    //    setup.get<TrackingComponentsRecord>().get(thePropagatorName,thePropagator);
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


TransientTrackingRecHit::ConstRecHitContainer
TrackTransformer::getTransientRecHits(const reco::TransientTrack& track) const {

  TransientTrackingRecHit::ConstRecHitContainer result;
  
  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit)
    if((*hit)->isValid())
      if ( (*hit)->geographicalId().det() == DetId::Tracker )
	result.push_back(theTrackerRecHitBuilder->build(&**hit));
      else if ( (*hit)->geographicalId().det() == DetId::Muon )
	result.push_back(theMuonRecHitBuilder->build(&**hit));
  
  return result;
}


/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformer::transform(const reco::Track& newTrack){
  
  const std::string metname = "Reco|TrackingTools|TrackTransformer";
  
  reco::TransientTrack track(newTrack,magneticField(),trackingGeometry());   
  
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit = getTransientRecHits(track);
  
  
  TrajectoryStateOnSurface firstTSOS;

  if(theRefitDirection == insideOut)
    firstTSOS = track.innermostMeasurementState();
  else{
    firstTSOS = track.outermostMeasurementState();
    reverse(recHitsForReFit.begin(),recHitsForReFit.end());
  }

  if(recHitsForReFit.size() < 2) return vector<Trajectory>();
  
  
  if(!firstTSOS.isValid()){
    LogDebug(metname)<<"Propagation error!"<<endl;
    return vector<Trajectory>();
  }
  
  TrajectorySeed seed;
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

