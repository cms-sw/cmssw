#include "RecoTracker/TkSeedGenerator/interface/SeedFromConsecutiveHits.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

using namespace std;

SeedFromConsecutiveHits::
SeedFromConsecutiveHits( const TrackingRecHit* outerHit, 
			 const TrackingRecHit* innerHit, 
			 const GlobalPoint& vertexPos,
			 const GlobalError& vertexErr,
			 const edm::EventSetup& iSetup,
			 const edm::ParameterSet& p){

  isValid_ = construct( outerHit,  innerHit, vertexPos, vertexErr,iSetup,p) ;
}

SeedFromConsecutiveHits:: SeedFromConsecutiveHits(
    const SeedingHitSet & ordered,
    const GlobalPoint& vertexPos,
    const GlobalError& vertexErr,
    const edm::EventSetup& es)
  : isValid_(false)
{
  const SeedingHitSet::Hits & hits = ordered.hits(); 
  if ( hits.size() < 2) return;

  // build initial helix and FTS
  const TransientTrackingRecHit::ConstRecHitPointer& tth1 = hits[0];
  const TransientTrackingRecHit::ConstRecHitPointer& tth2 = hits[1];
  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), GlobalPoint(0.,0.,0.),es);
  GlobalTrajectoryParameters kine = helix.stateAtVertex().parameters();
  float sinTheta = sin( kine.momentum().theta() );
  FreeTrajectoryState fts( kine, initialError( vertexPos, vertexErr, sinTheta));

  // get tracker
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get propagator
  edm::ESHandle<Propagator>  thePropagatorHandle;
  es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",thePropagatorHandle);
  const Propagator*  thePropagator = &(*thePropagatorHandle);

  // get updator
  KFUpdator     theUpdator;

  TrajectoryStateOnSurface updatedState;
  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit];
    TrajectoryStateOnSurface state = (iHit==0) ? 
        thePropagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : thePropagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return;

    const TransientTrackingRecHit::ConstRecHitPointer& tth = hits[iHit]; 
    
    TransientTrackingRecHit::RecHitPointer newtth = tth->clone(state);
    updatedState =  theUpdator.update(state, *newtth);

    _hits.push_back(newtth->hit()->clone());
  } 
  PTraj = boost::shared_ptr<PTrajectoryStateOnDet>(
    transformer.persistentState(updatedState, hit->geographicalId().rawId()) );

  isValid_ = true;
}




bool SeedFromConsecutiveHits::
construct( const TrackingRecHit* outerHit, 
	   const TrackingRecHit* innerHit, 
	   const GlobalPoint& vertexPos,
	   const GlobalError& vertexErr,
	   const edm::EventSetup& iSetup,
	   const edm::ParameterSet& p) 
{
  typedef TrajectoryStateOnSurface     TSOS;
  typedef TrajectoryMeasurement        TM;


  // get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  GlobalPoint inner = 
      tracker->idToDet(innerHit->geographicalId())->surface().toGlobal(innerHit->localPosition());
  GlobalPoint outer = 
      tracker->idToDet(outerHit->geographicalId())->surface().toGlobal(outerHit->localPosition());

  FastHelix helix(outer, inner, GlobalPoint(0.,0.,0.),iSetup);
  
  GlobalTrajectoryParameters kine = helix.stateAtVertex().parameters();
  float sinTheta = sin( kine.momentum().theta() );
  FreeTrajectoryState fts( kine, initialError( vertexPos, vertexErr, sinTheta));

  edm::ESHandle<Propagator>  thePropagatorHandle;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",thePropagatorHandle);
  const Propagator*  thePropagator = &(*thePropagatorHandle);



  KFUpdator     theUpdator;

  const TSOS innerState = 
      thePropagator->propagate(fts,tracker->idToDet(innerHit->geographicalId())->surface());
  if ( !innerState.isValid()) return false;


  //
  // get the transient builder
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = p.getParameter<std::string>("TTRHBuilder");  
  iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);


  intrhit=theBuilder.product()->build(innerHit);

  const TSOS innerUpdated= theUpdator.update( innerState,*intrhit);			      

  TSOS outerState = thePropagator->propagate( innerUpdated,
      tracker->idToDet(outerHit->geographicalId())->surface());
 
  if ( !outerState.isValid()) return false;
  
  outrhit=theBuilder.product()->build(outerHit);

  TSOS outerUpdated = theUpdator.update( outerState, *outrhit);
 
  _hits.push_back(innerHit->clone());
  _hits.push_back(outerHit->clone());

  PTraj = boost::shared_ptr<PTrajectoryStateOnDet>( 
      transformer.persistentState(outerUpdated, outerHit->geographicalId().rawId()) );

  return true;
}


CurvilinearTrajectoryError SeedFromConsecutiveHits::
   initialError( const GlobalPoint& vertexPos, const GlobalError& vertexErr, float sinTheta) 
{
  AlgebraicSymMatrix C(5,1);

  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr*sinTheta;

  return CurvilinearTrajectoryError(C);
}

