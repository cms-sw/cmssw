#include "SeedFitter.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {

  template <class T>
  inline T sqr(T t) {
    return t * t;
  }

}  // namespace

SeedFitter::SeedFitter(const edm::ParameterSet& cfg) : SeedFitter::SeedFitter(cfg, consumesCollector()) {}

SeedFitter::SeedFitter(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
    : eleSeedCollectionToken_(
          consumes<reco::ElectronSeedCollection>(cfg.getParameter<edm::InputTag>("eleSeedCollection"))),
      thePropagatorLabel_(cfg.getParameter<std::string>("propagator")),
      theBOFFMomentum_(cfg.getParameter<double>("SeedMomentumForBOFF")),
      theOriginTransverseErrorMultiplier_(cfg.getParameter<double>("OriginTransverseErrorMultiplier")),
      theMinOneOverPtError_(cfg.getParameter<double>("MinOneOverPtError")),
      TTRHBuilder_(cfg.getParameter<std::string>("TTRHBuilder")),
      mfName_(cfg.getParameter<std::string>("magneticField")),
      trackerGeometryESToken_(iC.esConsumes()),
      propagatorESToken_(iC.esConsumes(edm::ESInputTag("", thePropagatorLabel_))),
      magneticFieldESToken_(iC.esConsumes(edm::ESInputTag("", mfName_))),
      transientTrackingRecHitBuilderESToken_(iC.esConsumes(edm::ESInputTag("", TTRHBuilder_))) {
  produces<reco::ElectronSeedCollection>();
}

SeedFitter::~SeedFitter() {}

void SeedFitter::fillDescription(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("eleSeedCollection", edm::InputTag("hltEgammaFittedElectronPixelSeeds"));
  desc.add<std::string>("propagator", "PropagatorWithMaterialParabolicMf");
  desc.add<double>("SeedMomentumForBOFF", 5.0);
  desc.add<double>("OriginTransverseErrorMultiplier", 1.0);
  desc.add<double>("MinOneOverPtError", 1.0);
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.add<std::string>("magneticField", "ParabolicMf");

  descriptions.add("seedFitter", desc);
}

void SeedFitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  init(iSetup);

  edm::Handle<reco::ElectronSeedCollection> eleSeedCollection;
  iEvent.getByToken(eleSeedCollectionToken_, eleSeedCollection);
  const reco::ElectronSeedCollection& electronSeedsIn = *eleSeedCollection;

  auto electronSeedsOut = std::make_unique<reco::ElectronSeedCollection>();
  electronSeedsOut->reserve(electronSeedsIn.size());

  for (const reco::ElectronSeed& seed : electronSeedsIn) {
    makeSeed(*electronSeedsOut, seed);
  }

  iEvent.put(std::move(electronSeedsOut));
}

void SeedFitter::init(const edm::EventSetup& es) {
  trackerGeometry_ = &es.getData(trackerGeometryESToken_);
  propagator_ = &es.getData(propagatorESToken_);
  magneticField_ = &es.getData(magneticFieldESToken_);
  nomField_ = magneticField_->nominalValue();
  isBOFF_ = (0 == nomField_);

  TransientTrackingRecHitBuilder const* transientTrackingRecHitBuilder =
      &es.getData(transientTrackingRecHitBuilderESToken_);
  auto builder = (TkTransientTrackingRecHitBuilder const*)(transientTrackingRecHitBuilder);
  cloner_ = (*builder).cloner();
}

void SeedFitter::makeSeed(reco::ElectronSeedCollection& seedCollection, const reco::ElectronSeed& seed) {
  if (seed.nHits() < 2) {
    return;
  }

  GlobalTrajectoryParameters kine;
  initialKinematic(kine, seed);

  auto sin2Theta = kine.momentum().perp2() / kine.momentum().mag2();

  CurvilinearTrajectoryError error = initialError(sin2Theta);
  FreeTrajectoryState fts(kine, error);

  buildSeed(seedCollection, seed, fts);
}

void SeedFitter::initialKinematic(GlobalTrajectoryParameters& kine, const reco::ElectronSeed& seed) const {
  const TrackingRecHit& tth1 = *(seed.recHits().begin());
  const TrackingRecHit& tth2 = *(seed.recHits().begin() + 1);

  const GlobalPoint vertexPos;

  FastHelix helix(tth2.globalPosition(), tth1.globalPosition(), vertexPos, nomField_, magneticField_);
  if (helix.isValid()) {
    kine = helix.stateAtVertex();
  } else {
    GlobalVector initMomentum(tth2.globalPosition() - vertexPos);
    initMomentum *= (100.f / initMomentum.perp());
    kine = GlobalTrajectoryParameters(vertexPos, initMomentum, 1, magneticField_);
  }

  if UNLIKELY (isBOFF_ && (theBOFFMomentum_ > 0)) {
    kine = GlobalTrajectoryParameters(
        kine.position(), kine.momentum().unit() * theBOFFMomentum_, kine.charge(), magneticField_);
  }
}

CurvilinearTrajectoryError SeedFitter::initialError(float sin2Theta) const {
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit information.
  CurvilinearTrajectoryError newError;  // zeroed
  auto& C = newError.matrix();

  // FIXME: minC00. Prevent apriori uncertainty in 1/P from being too small, to avoid instabilities.
  // N.B. This parameter needs optimising ...
  // Probably OK based on quick study: KS 22/11/12.
  auto sin2th = sin2Theta;
  auto minC00 = sqr(theMinOneOverPtError_);
  C[0][0] = std::max(sin2th / sqr(/*region->ptMin()*/ 1.5f),
                     minC00);  // hardcocde region->ptMin() as 1.5 for now, as provided in hlt menu
  auto zErr =
      sqr(/*region->originZBound()*/ 12.5);  // hardcocde region->ptMin() as 12.5 for now, as provided in hlt menu
  auto transverseErr = sqr(
      theOriginTransverseErrorMultiplier_ *
      /*region->originRBound()*/ 0.2);  // hardcode region->originRBound() as 0.2 for now, as provided in the hlt menu
  C[1][1] = C[2][2] = 1.;               // no good reason. no bad reason....
  C[3][3] = transverseErr;
  C[4][4] = zErr * sin2th + transverseErr * (1.f - sin2th);

  return newError;
}

void SeedFitter::buildSeed(reco::ElectronSeedCollection& seedCollection,
                           const reco::ElectronSeed& seed,
                           const FreeTrajectoryState& fts) const {
  // get updator
  KFUpdator updator;

  // Now update initial state track using information from seed hits.

  TrajectoryStateOnSurface updatedState;
  edm::OwnVector<TrackingRecHit> seedHits;

  TrackingRecHit::id_type lastHitId = 0;

  for (unsigned int iHit = 0; iHit < seed.nHits(); iHit++) {
    const TrackingRecHit& hit = *(seed.recHits().begin() + iHit);
    TrajectoryStateOnSurface state =
        (iHit == 0) ? propagator_->propagate(fts, trackerGeometry_->idToDet(hit.geographicalId())->surface())
                    : propagator_->propagate(updatedState, trackerGeometry_->idToDet(hit.geographicalId())->surface());
    if (!state.isValid()) {
      return;
    }

    BaseTrackerRecHit const* tth = static_cast<BaseTrackerRecHit const*>(&hit);

    if (not tth) {
      return;
    }

    std::unique_ptr<BaseTrackerRecHit> newtth(refitHit(tth, state));

    if (not newtth) {
      return;
    }

    updatedState = updator.update(state, *newtth);
    if (!updatedState.isValid()) {
      return;
    }

    lastHitId = hit.geographicalId().rawId();
    seedHits.push_back(newtth.release());
  }

  PTrajectoryStateOnDet const& PTraj = trajectoryStateTransform::persistentState(updatedState, lastHitId);

  reco::ElectronSeed eleSeed(TrajectorySeed(PTraj, std::move(seedHits), alongMomentum));
  const reco::ElectronSeed::CaloClusterRef& caloClusRef(seed.caloCluster());
  eleSeed.setCaloCluster(caloClusRef);
  eleSeed.setNrLayersAlongTraj(seed.nrLayersAlongTraj());
  for (auto const& hitInfo : seed.hitInfo()) {
    eleSeed.addHitInfo(hitInfo);
  }

  seedCollection.emplace_back(eleSeed);
}

SeedingHitSet::RecHitPointer SeedFitter::refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                                  const TrajectoryStateOnSurface& state) const {
  return (SeedingHitSet::RecHitPointer)(cloner_(*hit, state));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SeedFitter);
