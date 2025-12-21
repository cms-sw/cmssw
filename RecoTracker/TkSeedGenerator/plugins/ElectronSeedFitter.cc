#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/Visibility.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace {

  template <class T>
  inline T sqr(T t) {
    return t * t;
  }

}  // namespace

class dso_hidden ElectronSeedFitter : public edm::stream::EDProducer<> {
public:
  ElectronSeedFitter(const edm::ParameterSet&);

  ~ElectronSeedFitter() override = default;

  static void fillDescription(edm::ConfigurationDescriptions& description);

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  // initialize the "event dependent state"
  void init(const edm::EventSetup& es);

  // make job
  // fill seedCollection with the "ElectronSeedCollection"
  void makeSeed(reco::ElectronSeedCollection& seedCollection, const reco::ElectronSeed& seed);

private:
  void initialKinematic(GlobalTrajectoryParameters& kine, const reco::ElectronSeed& seed) const;

  CurvilinearTrajectoryError initialError(float sin2Theta) const dso_hidden;

  void buildSeed(reco::ElectronSeedCollection& seedCollection,
                 const reco::ElectronSeed& seed,
                 const FreeTrajectoryState& fts) const dso_hidden;

  SeedingHitSet::RecHitPointer refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                        const TrajectoryStateOnSurface& state) const dso_hidden;

protected:
  const edm::EDGetTokenT<reco::ElectronSeedCollection> eleSeedCollectionToken_;

  const std::string propagatorLabel_;
  const float bOFFMomentum_;
  const float originTransverseErrorMultiplier_;
  const float minOneOverPtError_;

  TrackerGeometry const* trackerGeometry_;
  Propagator const* propagator_;
  MagneticField const* magneticField_;
  float nomField_ = 0.;
  bool isBOFF_ = false;
  const std::string ttrhBuilder_;
  const std::string mfName_;

  TkClonerImpl cloner_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryESToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorESToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldESToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderESToken_;
  const edm::EDPutTokenT<reco::ElectronSeedCollection> putToken_;
};

ElectronSeedFitter::ElectronSeedFitter(const edm::ParameterSet& cfg)
    : eleSeedCollectionToken_(
          consumes<reco::ElectronSeedCollection>(cfg.getParameter<edm::InputTag>("eleSeedCollection"))),
      propagatorLabel_(cfg.getParameter<std::string>("propagator")),
      bOFFMomentum_(cfg.getParameter<double>("SeedMomentumForBOFF")),
      originTransverseErrorMultiplier_(cfg.getParameter<double>("OriginTransverseErrorMultiplier")),
      minOneOverPtError_(cfg.getParameter<double>("MinOneOverPtError")),
      ttrhBuilder_(cfg.getParameter<std::string>("TTRHBuilder")),
      mfName_(cfg.getParameter<std::string>("magneticField")),
      trackerGeometryESToken_(esConsumes()),
      propagatorESToken_(esConsumes(edm::ESInputTag("", propagatorLabel_))),
      magneticFieldESToken_(esConsumes(edm::ESInputTag("", mfName_))),
      transientTrackingRecHitBuilderESToken_(esConsumes(edm::ESInputTag("", ttrhBuilder_))),
      putToken_{produces()} {}

void ElectronSeedFitter::fillDescription(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("eleSeedCollection", edm::InputTag("hltEgammaFittedElectronPixelSeeds"));
  desc.add<std::string>("propagator", "PropagatorWithMaterialParabolicMf");
  desc.add<double>("SeedMomentumForBOFF", 5.0);
  desc.add<double>("OriginTransverseErrorMultiplier", 1.0);
  desc.add<double>("MinOneOverPtError", 1.0);
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.add<std::string>("magneticField", "ParabolicMf");

  descriptions.addWithDefaultLabel(desc);
}

void ElectronSeedFitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  init(iSetup);

  const auto& electronSeedsIn = *iEvent.getHandle(eleSeedCollectionToken_);

  auto electronSeedsOut = std::make_unique<reco::ElectronSeedCollection>();
  electronSeedsOut->reserve(electronSeedsIn.size());

  for (const reco::ElectronSeed& seed : electronSeedsIn) {
    makeSeed(*electronSeedsOut, seed);
  }

  electronSeedsOut->shrink_to_fit();
  iEvent.put(putToken_, std::move(electronSeedsOut));
}

void ElectronSeedFitter::init(const edm::EventSetup& es) {
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

void ElectronSeedFitter::makeSeed(reco::ElectronSeedCollection& seedCollection, const reco::ElectronSeed& seed) {
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

void ElectronSeedFitter::initialKinematic(GlobalTrajectoryParameters& kine, const reco::ElectronSeed& seed) const {
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

  if UNLIKELY (isBOFF_ && (bOFFMomentum_ > 0)) {
    kine = GlobalTrajectoryParameters(
        kine.position(), kine.momentum().unit() * bOFFMomentum_, kine.charge(), magneticField_);
  }
}

CurvilinearTrajectoryError ElectronSeedFitter::initialError(float sin2Theta) const {
  // Set initial uncertainty on track parameters, using only P.V. constraint and no hit information.
  CurvilinearTrajectoryError newError;  // zeroed
  auto& C = newError.matrix();

  auto sin2th = sin2Theta;
  auto minC00 = sqr(minOneOverPtError_);
  C[0][0] = std::max(sin2th / sqr(1.5f), minC00);
  auto zErr = sqr(12.5);
  auto transverseErr = sqr(originTransverseErrorMultiplier_ * 0.2);
  C[1][1] = C[2][2] = 1.;
  C[3][3] = transverseErr;
  C[4][4] = zErr * sin2th + transverseErr * (1.f - sin2th);

  return newError;
}

void ElectronSeedFitter::buildSeed(reco::ElectronSeedCollection& seedCollection,
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

SeedingHitSet::RecHitPointer ElectronSeedFitter::refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                                          const TrajectoryStateOnSurface& state) const {
  return (SeedingHitSet::RecHitPointer)(cloner_(*hit, state));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronSeedFitter);
