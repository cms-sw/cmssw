#include "SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

namespace {

  template <class T> 
  inline
  T sqr( T t) {return t*t;}

}

SeedFromConsecutiveHitsCreator::~SeedFromConsecutiveHitsCreator(){}

void SeedFromConsecutiveHitsCreator::init(const TrackingRegion & iregion,
	  const edm::EventSetup& es,
	  const SeedComparitor *ifilter) {
  region = &iregion;
  filter = ifilter;
  // get tracker
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  // get propagator
  es.get<TrackingComponentsRecord>().get(thePropagatorLabel, propagatorHandle);
  // mag field
  es.get<IdealMagneticFieldRecord>().get(bfield);
  nomField = bfield->nominalValue();
  isBOFF = (0==nomField);  
}

void SeedFromConsecutiveHitsCreator::makeSeed(TrajectorySeedCollection & seedCollection,
					      const SeedingHitSet & hits) {
  if ( hits.size() < 2) return;

  GlobalTrajectoryParameters kine;
  if (!initialKinematic(kine, hits)) return;

  float sin2Theta = kine.momentum().perp2()/kine.momentum().mag2();

  CurvilinearTrajectoryError error = initialError(sin2Theta);
  FreeTrajectoryState fts(kine, error);

  buildSeed(seedCollection,hits,fts); 
}



bool SeedFromConsecutiveHitsCreator::initialKinematic(GlobalTrajectoryParameters & kine,
						      const SeedingHitSet & hits) const{

  TransientTrackingRecHit::ConstRecHitPointer tth1 = hits[0];
  TransientTrackingRecHit::ConstRecHitPointer tth2 = hits[1];
  const GlobalPoint& vertexPos = region->origin();

  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, nomField,&*bfield);
  if (helix.isValid()) {
    kine = helix.stateAtVertex();
  } else {
    GlobalVector initMomentum(tth2->globalPosition() - vertexPos);
    initMomentum *= (100./initMomentum.perp()); 
    kine = GlobalTrajectoryParameters(vertexPos, initMomentum, 1, &*bfield);
  } 

  if unlikely(isBOFF && (theBOFFMomentum > 0)) {
      kine = GlobalTrajectoryParameters(kine.position(),
					kine.momentum().unit() * theBOFFMomentum,
					kine.charge(),
					&*bfield);
  }
  return (filter ? filter->compatible(hits, kine, helix, *region) : true); 
}



CurvilinearTrajectoryError
SeedFromConsecutiveHitsCreator::initialError(float sin2Theta) const
{
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit
  // information.
  AlgebraicSymMatrix55 C = ROOT::Math::SMatrixIdentity();

// FIXME: minC00. Prevent apriori uncertainty in 1/P from being too small, 
// to avoid instabilities.
// N.B. This parameter needs optimising ...
  // Probably OK based on quick study: KS 22/11/12.
  float sin2th = sin2Theta;
  float minC00 = sqr(theMinOneOverPtError);
  C[0][0] = std::max(sin2th/sqr(region->ptMin()), minC00);
  float zErr = sqr(region->originZBound());
  float transverseErr = sqr(theOriginTransverseErrorMultiplier*region->originRBound());
  C[3][3] = transverseErr;
  C[4][4] = zErr*sin2th + transverseErr*(1.f-sin2th);

  return CurvilinearTrajectoryError(C);
}

void SeedFromConsecutiveHitsCreator::buildSeed(
    TrajectorySeedCollection & seedCollection,
    const SeedingHitSet & hits,
    const FreeTrajectoryState & fts) const
{
  const Propagator*  propagator = &(*propagatorHandle);
  
  // get updator
  KFUpdator  updator;
  
  // Now update initial state track using information from seed hits.
  
  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;
  
  const TrackingRecHit* hit = 0;
  for ( unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit]->hit();
    TrajectoryStateOnSurface state = (iHit==0) ? 
      propagator->propagate(fts,tracker->idToDet(hit->geographicalId())->surface())
      : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid()) return;
    
    TransientTrackingRecHit::ConstRecHitPointer const &  tth = hits[iHit]; 
    
    TransientTrackingRecHit::RecHitPointer const & newtth = refitHit( tth, state);
    
    if (!checkHit(state,newtth)) return;

    updatedState =  updator.update(state, *newtth);
    if (!updatedState.isValid()) return;
    
    seedHits.push_back(newtth->hit()->clone());

  } 

  
  PTrajectoryStateOnDet const & PTraj = 
    trajectoryStateTransform::persistentState(updatedState, hit->geographicalId().rawId());
  TrajectorySeed seed(PTraj,std::move(seedHits),alongMomentum); 
  if ( !filter || filter->compatible(seed)) seedCollection.push_back(seed);

}

TransientTrackingRecHit::RecHitPointer SeedFromConsecutiveHitsCreator::refitHit(
      const TransientTrackingRecHit::ConstRecHitPointer &hit, 
      const TrajectoryStateOnSurface &state) const
{
  return hit->clone(state);
}

bool 
SeedFromConsecutiveHitsCreator::checkHit(
      const TrajectoryStateOnSurface &tsos,
      const TransientTrackingRecHit::ConstRecHitPointer &hit) const 
{ 
    return (filter ? filter->compatible(tsos,hit) : true); 
}

