#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
//#include "TrackingTools/GeomPropagators/interface/Propagator.h"

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

  theFitterName = parameterSet.getParameter<string>("Fitter");  
  theSmootherName = parameterSet.getParameter<string>("Smoother");  
  
  //thePropagatorName = parameterSet.getParameter<string>("Propagator");

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
  
  TransientTrackingRecHit::ConstRecHitContainer transientRecHits = getTransientRecHits(track);
  
  // The outermost state is made of the combination of the most external rechit and the 
  // state coming from the in-out refit
  TrajectoryStateOnSurface outerTSOS = track.outermostMeasurementState();
  
  TransientTrackingRecHit::ConstRecHitContainer recHitsForReFit;
  
  // copy(transientRecHits.begin(),transientRecHits.end()-1,back_inserter(recHitsForReFit));
  copy(transientRecHits.begin(),transientRecHits.end(),back_inserter(recHitsForReFit));
  reverse(recHitsForReFit.begin(),recHitsForReFit.end());
  
  if(recHitsForReFit.size() < 2) return vector<Trajectory>();
  
  // In order to avoid to refit two times the first hit (the outerTSOS already contains its information)
  // the first propagation is done by hand
  TrajectoryStateOnSurface lastBOpredictedTSOS = 
    // thePropagator->propagate(outerTSOS,recHitsForReFit.front()->det()->surface());
    outerTSOS;
  
  if(!lastBOpredictedTSOS.isValid()){
    LogDebug(metname)<<"Propagation error!"<<endl;
    return vector<Trajectory>();
  }
  
  TrajectorySeed seed;
  vector<Trajectory> trajectories = theFitter->fit(seed,recHitsForReFit,lastBOpredictedTSOS);
  
  
  if(!trajectories.size()){
    LogDebug(metname)<<"No Track refitted!"<<endl;
    return vector<Trajectory>();
  }
  
  Trajectory trajectoryBW = trajectories.front();
    
  vector<Trajectory> trajectoriesSM = theSmoother->trajectories(trajectoryBW);
  
  if(!trajectoriesSM.size()){
    LogDebug(metname)<<"No Track smoothed!"<<endl;
    return vector<Trajectory>();
    // FIXME! Maybe should be:
    // return trajectories;
  }
  
  return trajectoriesSM;
}

