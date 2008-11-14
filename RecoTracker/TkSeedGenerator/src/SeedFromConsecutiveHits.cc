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
    std::string propagatorLabel,
    bool useFastHelix,
    double aBOFFMomentum)
  : isValid_(false)
{
  const SeedingHitSet::Hits & hits = ordered.hits(); 
  if ( hits.size() < 2) return;

  edm::ESHandle<MagneticField> bfield;
  es.get<IdealMagneticFieldRecord>().get(bfield);
  bool isBOFF = ( std::abs(bfield->inTesla(GlobalPoint(0,0,0)).z()) < 1e-3 );

  // build initial track estimate (ideally using beam spot constraint
  // and no significant information from any Tracker hits)
  GlobalTrajectoryParameters kine;
  const TransientTrackingRecHit::ConstRecHitPointer& tth1 = hits[0];
  const TransientTrackingRecHit::ConstRecHitPointer& tth2 = hits[1];

  if (useFastHelix) {
    // Assume initial state is helix passing through beam spot and innermost
    // two seed hits.
    FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, es);
    kine = helix.stateAtVertex().parameters();

    if (isBOFF && (aBOFFMomentum > 0)) {
      kine = GlobalTrajectoryParameters(kine.position(),
                                        kine.momentum().unit() * aBOFFMomentum,
                                        kine.charge(),
                                        &*bfield);
    }
  } else {
    // Assume initial state is straight line passing through beam spot
    // with direction given by innermost two seed hits (with big uncertainty)
    GlobalVector initMomentum(tth2->globalPosition() - tth1->globalPosition());
    double rescale = 1000./initMomentum.perp(); 
    initMomentum *= rescale; // set to approximately infinite momentum
    TrackCharge q = 1; // irrelevant, since infinite momentum
    kine = GlobalTrajectoryParameters(vertexPos, initMomentum, q, &*bfield);
  }

  // Make corresponding FTS, including estimate of current track param uncertainties.
  float sinTheta = sin( kine.momentum().theta() );
  FreeTrajectoryState fts( kine, initialError( vertexPos, vertexErr, sinTheta, ptMin));

  // get tracker
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  
  // get propagator
  edm::ESHandle<Propagator>  propagatorHandle;
  es.get<TrackingComponentsRecord>().get(propagatorLabel, propagatorHandle);
  const Propagator*  propagator = &(*propagatorHandle);

  // get updator
  KFUpdator  updator;

  // Now update initial state track using information from seed hits.

  TrajectoryStateOnSurface updatedState;
  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit];
    TrajectoryStateOnSurface state = (iHit==0) ? 
        propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return;

    const TransientTrackingRecHit::ConstRecHitPointer& tth = hits[iHit]; 
    
    TransientTrackingRecHit::RecHitPointer newtth = tth->clone(state);
    updatedState =  updator.update(state, *newtth);

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
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit
  // information.

  AlgebraicSymMatrix C(5,1);

  float sin2th = sqr(sinTheta);
  C[0][0] = sin2th/sqr(ptMin);
  float zErr = vertexErr.czz();
  float transverseErr = vertexErr.cxx(); // assume equal cxx cyy
  C[3][3] = transverseErr;
  C[4][4] = zErr*sin2th + transverseErr*(1-sin2th);

  return CurvilinearTrajectoryError(C);
}
