#include "SeedFromConsecutiveHits.h"
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
    const edm::EventSetup& es, const edm::ParameterSet& p)
  : isValid_(false)
{
  const SeedingHitSet::Hits & hits = ordered.hits(); 

  //
  // FIXME - clearly temporary !!!!
  //

  if (hits.size() >=2) {
    const TrackingRecHit* innerHit = hits[0].RecHit();
    const TrackingRecHit* outerHit = hits[1].RecHit();
    isValid_ = construct( outerHit,  innerHit, vertexPos, vertexErr, es, p);
  }
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

  FreeTrajectoryState fts( 
      helix.stateAtVertex().parameters(), initialError( outerHit, innerHit, vertexPos, vertexErr));

    
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
initialError( const TrackingRecHit* outerHit,
	      const TrackingRecHit* innerHit,
	      const GlobalPoint& vertexPos,
	      const GlobalError& vertexErr) 
{
  AlgebraicSymMatrix C(5,1);

  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr;

  return CurvilinearTrajectoryError(C);
}

