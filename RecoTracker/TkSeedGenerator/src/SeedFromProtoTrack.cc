#include "RecoTracker/TkSeedGenerator/interface/SeedFromProtoTrack.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SeedFromProtoTrack::Config::Config(edm::ConsumesCollector iC)
    : trackerToken_(iC.esConsumes()),
      propagatorToken_(iC.esConsumes(edm::ESInputTag("", "PropagatorWithMaterial"))),
      fieldToken_(iC.esConsumes()) {}

SeedFromProtoTrack::SeedFromProtoTrack(const Config& config,
                                       const reco::Track& proto,
                                       const SeedingHitSet& hits,
                                       const edm::EventSetup& es)
    : theValid(true) {
  for (unsigned int i = 0, n = hits.size(); i < n; ++i) {
    const TrackingRecHit* trh = hits[i]->hit();
    theHits.push_back(trh->clone());
  }
  init(config, proto, es);
}

SeedFromProtoTrack::SeedFromProtoTrack(const Config& config, const reco::Track& proto, const edm::EventSetup& es)
    : theValid(true) {
  const TrackingRecHit* hit = nullptr;
  for (unsigned int iHit = 0, nHits = proto.recHitsSize(); iHit < nHits; ++iHit) {
    TrackingRecHitRef refHit = proto.recHit(iHit);
    hit = &(*refHit);
    theHits.push_back(hit->clone());
  }
  init(config, proto, es);
}

void SeedFromProtoTrack::init(const Config& config, const reco::Track& proto, const edm::EventSetup& es) {
  TrackerGeometry const& tracker = es.getData(config.trackerToken_);

  const Propagator* propagator = &es.getData(config.propagatorToken_);

  const MagneticField* field = &es.getData(config.fieldToken_);

  const reco::TrackBase::Point& vtx = proto.referencePoint();
  const reco::TrackBase::Vector& mom = proto.momentum();
  GlobalTrajectoryParameters gtp(
      GlobalPoint(vtx.x(), vtx.y(), vtx.z()), GlobalVector(mom.x(), mom.y(), mom.z()), proto.charge(), field);

  CurvilinearTrajectoryError err = proto.covariance();

  FreeTrajectoryState fts(gtp, err);

  const TrackingRecHit& lastHit = theHits.back();

  TrajectoryStateOnSurface outerState =
      propagator->propagate(fts, tracker.idToDet(lastHit.geographicalId())->surface());

  if (!outerState.isValid()) {
    const Surface& surface = tracker.idToDet(lastHit.geographicalId())->surface();
    edm::LogError("SeedFromProtoTrack") << " was trying to create a seed from:\n"
                                        << fts << "\n propagating to: " << std::hex << lastHit.geographicalId().rawId()
                                        << std::dec << ' ' << surface.position();
    theValid = false;
    return;
  }
  theValid = true;

  thePTraj = trajectoryStateTransform::persistentState(outerState, lastHit.geographicalId().rawId());
}

TrajectorySeed SeedFromProtoTrack::trajectorySeed() const {
  return TrajectorySeed(trajectoryState(), hits(), direction());
}
