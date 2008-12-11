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
template <class T> T sqr( T t) {return t*t;}

SeedFromConsecutiveHits:: SeedFromConsecutiveHits(
    const SeedingHitSet & ordered,
    const GlobalPoint& vertexPos,
    const GlobalError& vertexErr,
    const edm::EventSetup& es,
    float ptMin,
    double theBOFFMomentum)
  : isValid_(false)
{
  const SeedingHitSet::Hits & hits = ordered.hits(); 
  if ( hits.size() < 2) return;

  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);
  bool isBOFF = ( std::abs(bfield->inTesla(GlobalPoint(0,0,0)).z()) < 1e-3 );

  // build initial helix and FTS
  const TransientTrackingRecHit::ConstRecHitPointer& tth1 = hits[0];
  const TransientTrackingRecHit::ConstRecHitPointer& tth2 = hits[1];
  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, es);
  GlobalTrajectoryParameters kine = helix.stateAtVertex().parameters();

  if (isBOFF && (theBOFFMomentum > 0)) {
    kine = GlobalTrajectoryParameters(kine.position(),
                                      kine.momentum().unit() * theBOFFMomentum,
                                      kine.charge(),
                                      &*bfield);
  }

  float sinTheta = sin( kine.momentum().theta() );
  FreeTrajectoryState fts( kine, initialError( vertexPos, vertexErr, sinTheta, ptMin));

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



CurvilinearTrajectoryError SeedFromConsecutiveHits::
   initialError( const GlobalPoint& vertexPos, const GlobalError& vertexErr, 
                 float sinTheta, float ptMin) 
{
  AlgebraicSymMatrix C(5,1);

  float sin2th = sqr(sinTheta);
  float minC00 = 1.0;
  C[0][0] = std::max(sin2th/sqr(ptMin), minC00);
  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy 
  C[3][3] = transverseErr;
  C[4][4] = zErr*sin2th + transverseErr*(1-sin2th);

  return CurvilinearTrajectoryError(C);
}

