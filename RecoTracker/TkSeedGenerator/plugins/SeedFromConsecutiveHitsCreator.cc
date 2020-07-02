#include "SeedFromConsecutiveHitsCreator.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {

  template <class T>
  inline T sqr(T t) {
    return t * t;
  }

}  // namespace

SeedFromConsecutiveHitsCreator::~SeedFromConsecutiveHitsCreator() {}

void SeedFromConsecutiveHitsCreator::fillDescriptions(edm::ParameterSetDescription& desc) {
  desc.add<std::string>("propagator", "PropagatorWithMaterialParabolicMf");
  desc.add<double>("SeedMomentumForBOFF", 5.0);
  desc.add<double>("OriginTransverseErrorMultiplier", 1.0);
  desc.add<double>("MinOneOverPtError", 1.0);
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.add<std::string>("magneticField", "ParabolicMf");
  desc.add<bool>("forceKinematicWithRegionDirection", false);
}

void SeedFromConsecutiveHitsCreator::init(const TrackingRegion& iregion,
                                          const edm::EventSetup& es,
                                          const SeedComparitor* ifilter) {
  region = &iregion;
  filter = ifilter;
  // get tracker
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  // get propagator
  es.get<TrackingComponentsRecord>().get(thePropagatorLabel, propagatorHandle);
  // mag field
  es.get<IdealMagneticFieldRecord>().get(mfName_, bfield);
  //  edm::ESInputTag mfESInputTag(mfName_);
  //  es.get<IdealMagneticFieldRecord>().get(mfESInputTag, bfield);
  nomField = bfield->nominalValue();
  isBOFF = (0 == nomField);

  edm::ESHandle<TransientTrackingRecHitBuilder> builderH;
  es.get<TransientRecHitRecord>().get(TTRHBuilder, builderH);
  auto builder = (TkTransientTrackingRecHitBuilder const*)(builderH.product());
  cloner = (*builder).cloner();
}

void SeedFromConsecutiveHitsCreator::makeSeed(TrajectorySeedCollection& seedCollection, const SeedingHitSet& hits) {
  if (hits.size() < 2)
    return;

  GlobalTrajectoryParameters kine;
  if (!initialKinematic(kine, hits))
    return;

  auto sin2Theta = kine.momentum().perp2() / kine.momentum().mag2();

  CurvilinearTrajectoryError error = initialError(sin2Theta);
  FreeTrajectoryState fts(kine, error);

  if (region->direction().x() != 0 &&
      forceKinematicWithRegionDirection_)  // a direction was given, check if it is an etaPhi region
  {
    const RectangularEtaPhiTrackingRegion* etaPhiRegion = dynamic_cast<const RectangularEtaPhiTrackingRegion*>(region);
    if (etaPhiRegion) {
      //the following completely reset the kinematics, perhaps it makes no sense and newKine=kine would do better
      GlobalVector direction = region->direction() / region->direction().mag();
      GlobalVector momentum = direction * fts.momentum().mag();
      GlobalPoint position = region->origin() + 5 * direction;
      GlobalTrajectoryParameters newKine(position, momentum, fts.charge(), &fts.parameters().magneticField());

      auto ptMin = region->ptMin();
      CurvilinearTrajectoryError newError;  //zeroed
      auto& C = newError.matrix();
      constexpr float minC00 = 0.4f;
      C[0][0] = std::max(sin2Theta / sqr(ptMin), minC00);
      auto zErr = sqr(region->originZBound());
      auto transverseErr = sqr(region->originRBound());  // assume equal cxx cyy
      auto twiceDeltaLambda =
          std::atan(etaPhiRegion->tanLambdaRange().first) - std::atan(etaPhiRegion->tanLambdaRange().second);
      auto twiceDeltaPhi = etaPhiRegion->phiMargin().right() + etaPhiRegion->phiMargin().left();
      C[1][1] = twiceDeltaLambda * twiceDeltaLambda;  //2 sigma of what given in input
      C[2][2] = twiceDeltaPhi * twiceDeltaPhi;
      C[3][3] = transverseErr;
      C[4][4] = zErr * sin2Theta + transverseErr * (1.f - sin2Theta);
      fts = FreeTrajectoryState(newKine, newError);
    }
  }

  buildSeed(seedCollection, hits, fts);
}

bool SeedFromConsecutiveHitsCreator::initialKinematic(GlobalTrajectoryParameters& kine,
                                                      const SeedingHitSet& hits) const {
  SeedingHitSet::ConstRecHitPointer tth1 = hits[0];
  SeedingHitSet::ConstRecHitPointer tth2 = hits[1];

  const GlobalPoint& vertexPos = region->origin();

  FastHelix helix(tth2->globalPosition(), tth1->globalPosition(), vertexPos, nomField, &*bfield);
  if (helix.isValid()) {
    kine = helix.stateAtVertex();
  } else {
    GlobalVector initMomentum(tth2->globalPosition() - vertexPos);
    initMomentum *= (100.f / initMomentum.perp());
    kine = GlobalTrajectoryParameters(vertexPos, initMomentum, 1, &*bfield);
  }

  if UNLIKELY (isBOFF && (theBOFFMomentum > 0)) {
    kine =
        GlobalTrajectoryParameters(kine.position(), kine.momentum().unit() * theBOFFMomentum, kine.charge(), &*bfield);
  }
  return (filter ? filter->compatible(hits, kine, helix) : true);
}

CurvilinearTrajectoryError SeedFromConsecutiveHitsCreator::initialError(float sin2Theta) const {
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit
  // information.
  CurvilinearTrajectoryError newError;  // zeroed
  auto& C = newError.matrix();

  // FIXME: minC00. Prevent apriori uncertainty in 1/P from being too small,
  // to avoid instabilities.
  // N.B. This parameter needs optimising ...
  // Probably OK based on quick study: KS 22/11/12.
  auto sin2th = sin2Theta;
  auto minC00 = sqr(theMinOneOverPtError);
  C[0][0] = std::max(sin2th / sqr(region->ptMin()), minC00);
  auto zErr = sqr(region->originZBound());
  auto transverseErr = sqr(theOriginTransverseErrorMultiplier * region->originRBound());
  C[1][1] = C[2][2] = 1.;  // no good reason. no bad reason....
  C[3][3] = transverseErr;
  C[4][4] = zErr * sin2th + transverseErr * (1.f - sin2th);

  return newError;
}

void SeedFromConsecutiveHitsCreator::buildSeed(TrajectorySeedCollection& seedCollection,
                                               const SeedingHitSet& hits,
                                               const FreeTrajectoryState& fts) const {
  const Propagator* propagator = &(*propagatorHandle);

  // get updator
  KFUpdator updator;

  // Now update initial state track using information from seed hits.

  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;

  const TrackingRecHit* hit = nullptr;
  for (unsigned int iHit = 0; iHit < hits.size(); iHit++) {
    hit = hits[iHit]->hit();
    TrajectoryStateOnSurface state =
        (iHit == 0) ? propagator->propagate(fts, tracker->idToDet(hit->geographicalId())->surface())
                    : propagator->propagate(updatedState, tracker->idToDet(hit->geographicalId())->surface());
    if (!state.isValid())
      return;

    SeedingHitSet::ConstRecHitPointer tth = hits[iHit];

    std::unique_ptr<BaseTrackerRecHit> newtth(refitHit(tth, state));

    if (!checkHit(state, &*newtth))
      return;

    updatedState = updator.update(state, *newtth);
    if (!updatedState.isValid())
      return;

    seedHits.push_back(newtth.release());
  }

  if (!hit)
    return;

  PTrajectoryStateOnDet const& PTraj =
      trajectoryStateTransform::persistentState(updatedState, hit->geographicalId().rawId());
  seedCollection.emplace_back(PTraj, std::move(seedHits), alongMomentum);
}

SeedingHitSet::RecHitPointer SeedFromConsecutiveHitsCreator::refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                                                      const TrajectoryStateOnSurface& state) const {
  return (SeedingHitSet::RecHitPointer)(cloner(*hit, state));
}

bool SeedFromConsecutiveHitsCreator::checkHit(const TrajectoryStateOnSurface& tsos,
                                              SeedingHitSet::ConstRecHitPointer hit) const {
  return (filter ? filter->compatible(tsos, hit) : true);
}
