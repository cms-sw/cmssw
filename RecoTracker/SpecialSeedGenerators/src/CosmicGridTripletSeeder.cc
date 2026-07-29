#include <Rtypes.h>
#include <RtypesCore.h>
#include <TAttLine.h>
#include <TAttMarker.h>
#include <TEllipse.h>
#include <cmath>
#include <memory>
#include <set>
#include <unordered_set>
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoTracker/PixelSeeding/interface/OrderedHitTriplet.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicGridTripletSeeder.h"

CosmicGridTripletSeeder::CosmicGridTripletSeeder(const edm::ParameterSet& iConfig)
    : vectorHitsToken_(mayConsume<VectorHitCollection>(iConfig.getUntrackedParameter<edm::InputTag>("vectorHits"))),
      otRecHitsToken_(
          mayConsume<Phase2TrackerRecHit1DCollectionNew>(iConfig.getUntrackedParameter<edm::InputTag>("OTRecHits"))),
      matchedStripHitsToken_(mayConsume<SiStripMatchedRecHit2DCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("matchedStripHits"))),
      rPhiHitsToken_(mayConsume<SiStripRecHit2DCollection>(iConfig.getUntrackedParameter<edm::InputTag>("rPhiHits"))),
      pixelRecHitsToken_(consumes(iConfig.getUntrackedParameter<edm::InputTag>("PixelRecHits"))),
      magfieldToken_(esConsumes(iConfig.getParameter<edm::ESInputTag>("MagneticFieldRecord"))),
      trackerToken_(esConsumes()),
      ttrhBuilderToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("TTRHBuilder")))),
      writeTriplets_(iConfig.getParameter<bool>("writeTriplets")),
      nGridBinsX_(iConfig.getParameter<int>("nGridBinsX")),
      nGridBinsY_(iConfig.getParameter<int>("nGridBinsY")),
      nGridBinsZ_(iConfig.getParameter<int>("nGridBinsZ")),
      gridXmin_(iConfig.getParameter<double>("gridXmin")),
      gridXmax_(iConfig.getParameter<double>("gridXmax")),
      gridYmin_(iConfig.getParameter<double>("gridYmin")),
      gridYmax_(iConfig.getParameter<double>("gridYmax")),
      gridZmin_(iConfig.getParameter<double>("gridZmin")),
      gridZmax_(iConfig.getParameter<double>("gridZmax")),
      maxTripsPerSP_(iConfig.getParameter<int>("maxTripsPerSP")),
      maxSeedsPerSP_(iConfig.getParameter<int>("maxSeedsPerSP")),
      slopeCompatForVH_(iConfig.getParameter<double>("slopeCompatForVH")),
      tryBothDirections_(iConfig.getParameter<bool>("tryBothDirections")) {
  produces<TrajectorySeedCollection>();
  if (writeTriplets_) {
    produces<edm::OwnVector<TrackingRecHit>>();
  }
}

void CosmicGridTripletSeeder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("vectorHits", edm::InputTag("siPhase2VectorHits:accepted"));
  desc.addUntracked<edm::InputTag>("OTRecHits", edm::InputTag("siPhase2RecHits"));
  desc.addUntracked<edm::InputTag>("matchedStripHits", edm::InputTag("siStripMatchedRecHits", "matchedRecHit"));
  desc.addUntracked<edm::InputTag>("rPhiHits", edm::InputTag("siStripMatchedRecHits", "rphiRecHit"));
  desc.addUntracked<edm::InputTag>("PixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.add<std::string>("TTRHBuilder", "WithTrackAngle");
  desc.add<edm::ESInputTag>("MagneticFieldRecord", edm::ESInputTag("", ""));
  desc.add<bool>("writeTriplets", false);
  desc.add<int>("nGridBinsX", 1);
  desc.add<int>("nGridBinsY", 1);
  desc.add<int>("nGridBinsZ", 1);
  desc.add<double>("gridXmin", -120);
  desc.add<double>("gridXmax", 120);
  desc.add<double>("gridYmin", -120);
  desc.add<double>("gridYmax", 120);
  desc.add<double>("gridZmin", -280);
  desc.add<double>("gridZmax", 280);
  desc.add<int>("maxTripsPerSP", 10);
  desc.add<int>("maxSeedsPerSP", 8);
  desc.add<double>("slopeCompatForVH", 0.15);
  desc.add<bool>("tryBothDirections", true);
  descriptions.addWithDefaultLabel(desc);
}

CosmicGridTripletSeeder::TripletSeederEventState CosmicGridTripletSeeder::initEventState(
    const edm::EventSetup& c) const {
  auto magfield = &c.getData(magfieldToken_);
  auto tracker = &c.getData(trackerToken_);
  auto cloner = dynamic_cast<TkTransientTrackingRecHitBuilder const&>(c.getData(ttrhBuilderToken_)).cloner();
  constexpr double g_muonMass = 0.1057;
  return TripletSeederEventState{
      CartesianSeedingGrid{
          nGridBinsX_, gridXmin_, gridXmax_, nGridBinsY_, gridYmin_, gridYmax_, nGridBinsZ_, gridZmin_, gridZmax_},
      {},
      magfield,
      tracker,
      cloner,
      std::make_unique<KFUpdator>(),
      std::make_unique<PropagatorWithMaterial>(alongMomentum, g_muonMass, magfield),
      std::make_unique<PropagatorWithMaterial>(oppositeToMomentum, g_muonMass, magfield)};
}

bool CosmicGridTripletSeeder::tripletOk(const TripletSeederEventState& state, const OrderedHitTriplet& trip) const {
  // get the positions for triplet shape cleaning
  Global3DPoint top = trip.inner()->globalPosition();
  Global3DPoint middle = trip.middle()->globalPosition();
  Global3DPoint bottom = trip.outer()->globalPosition();

  // veto "zigzag" in z
  if ((top.z() - middle.z()) * (middle.z() - bottom.z()) < 0 && std::abs(top.z() - bottom.z()) > 1.0)
    return false;

  // veto "zigzag" in x
  if ((top.x() - middle.x()) * (middle.x() - bottom.x()) < 0 && std::abs(top.x() - bottom.x()) > 2.0)
    return false;

  // x-y slope consistency checks

  // get the local x-y slopes between pairs of consecutive SP
  double dxy_t = (top.x() - middle.x()) / (top.y() - middle.y());
  double dxy_b = (middle.x() - bottom.x()) / (middle.y() - bottom.y());
  // center point estimated as mean between the adjacent segments
  double dxy_c = 0.5 * (dxy_t + dxy_b);
  double dxy_vh_c = 0;
  double dxy_vh_b = 0;
  double dxy_vh_t = 0;

  // and, for vector hits (only used in the barrel), the measured slopes
  // on the hit itself
  const VectorHit* topVH = dynamic_cast<const VectorHit*>(trip.inner());
  if (topVH)
    dxy_vh_t = topVH->globalDirectionVH().x() / topVH->globalDirectionVH().y();

  const VectorHit* centerVH = dynamic_cast<const VectorHit*>(trip.middle());
  if (centerVH)
    dxy_vh_c = centerVH->globalDirectionVH().x() / centerVH->globalDirectionVH().y();

  const VectorHit* bottomVH = dynamic_cast<const VectorHit*>(trip.outer());
  if (bottomVH)
    dxy_vh_b = bottomVH->globalDirectionVH().x() / bottomVH->globalDirectionVH().y();

  // now we require consistency between the VH and the estimate from neighbouring SP
  // for all cases where these are well-defined
  if (topVH && std::abs(dxy_vh_t - dxy_t) > slopeCompatForVH_ * std::abs(std::max(dxy_vh_t, dxy_t)))
    return false;
  if (centerVH && std::abs(dxy_vh_c - dxy_c) > slopeCompatForVH_ * std::abs(std::max(dxy_vh_c, dxy_c)))
    return false;
  if (bottomVH && std::abs(dxy_vh_b - dxy_b) > slopeCompatForVH_ * std::abs(std::max(dxy_vh_b, dxy_b)))
    return false;

  // check curvature estimate
  if (FastCircle(top, middle, bottom).rho() < 0.5 * ptMin_ / (0.003 * state.magfield->nominalValue())) {
    return false;
  }
  return true;
}

void CosmicGridTripletSeeder::produce(edm::Event& e, const edm::EventSetup& c) {
  // setup the event state object
  TripletSeederEventState state = initEventState(c);

  // populate the seeding grid
  populateGrid(e, state);

  // now form triplets
  std::vector<OrderedHitTriplet> triplets;
  formTriplets(state, triplets);

  // book the trajectory seed collection
  auto output = std::make_unique<TrajectorySeedCollection>();

  // and fit the triplets
  fitTriplets(state, triplets, *output);

  // put our output on the store
  e.put(std::move(output));
  // optional triplet writing for debugging / perf validation
  if (writeTriplets_) {
    auto outtriplets = std::make_unique<edm::OwnVector<TrackingRecHit>>();
    for (auto& trip : triplets) {
      outtriplets->push_back(trip.inner()->cloneHit());
      outtriplets->push_back(trip.middle()->cloneHit());
      outtriplets->push_back(trip.outer()->cloneHit());
    }
    e.put(std::move(outtriplets));
  }
}

void CosmicGridTripletSeeder::populateGrid(const edm::Event& iEvent,
                                           CosmicGridTripletSeeder::TripletSeederEventState& state) const {
  edm::Handle<VectorHitCollection> vectorHits;
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> otHitCollection;
  edm::Handle<SiStripMatchedRecHit2DCollection> matchedStripHits;
  edm::Handle<SiStripRecHit2DCollection> rPhiStripHits;
  edm::Handle<SiPixelRecHitCollection> pixelHitCollection;

  /// Treat the hit collections as optional, so that we can
  /// run with both the Run-3 and the Phase-2 detectors.
  bool hasOT = iEvent.getByToken(otRecHitsToken_, otHitCollection);
  bool hasVec = iEvent.getByToken(vectorHitsToken_, vectorHits);
  bool hasMatchedStrips = iEvent.getByToken(matchedStripHitsToken_, matchedStripHits);
  bool hasRphiStrips = iEvent.getByToken(rPhiHitsToken_, rPhiStripHits);

  /// Pixels should be around in all detector geometries.
  bool hasPixels = iEvent.getByToken(pixelRecHitsToken_, pixelHitCollection);

  std::set<const TrackingRecHit*> recHitsSeen{};
  std::vector<const VectorHit*> vhSeen{};

  /// step 1: Add the vector hits, and remember all raw hits associated to them
  if (hasVec) {
    LogDebug("CosmicGridTrupletSeeder") << "have Vec hits ";
    for (auto detSet : *vectorHits) {
      for (const VectorHit& vh : detSet) {
        // Phase-2 endcap vector hits currently have directional information
        // that can not be used directly - hence we skip these.
        if (state.tracker->idToDet(vh.geographicalId())->subDetector() == GeomDetEnumerators::SubDetector::P2OTEC) {
          continue;
        }
        state.grid.addHit(&vh);
        vhSeen.push_back(&vh);
      }
    }
  }

  // Step 2 for phase-2 detector

  // Add remaining OT hits, excluding those already on vector hits
  if (hasOT) {
    LogDebug("CosmicGridTrupletSeeder") << "have OT hits ";
    for (auto ds : *otHitCollection) {
      for (const auto& otHit : ds) {
        bool unique = true;
        for (const auto* vh : vhSeen) {
          if (vh->sharesInput(&otHit, TrackingRecHit::some)) {
            unique = false;
            state.vhConstituents[vh].push_back(&otHit);
            break;
          }
        }
        if (!unique) {
          continue;
        }
        state.grid.addHit(&otHit);
        recHitsSeen.insert(&otHit);
      }
    }
  }

  // Step 2 for a phase-1 detector (two hit collections)

  // phase-1 matched strip hits
  if (hasMatchedStrips) {
    LogDebug("CosmicGridTrupletSeeder") << "have matched hits ";
    for (auto ds : *matchedStripHits) {
      for (const auto& stripHit : ds) {
        bool unique = true;
        for (const auto* vh : vhSeen) {
          if (vh->sharesInput(&stripHit, TrackingRecHit::some)) {
            unique = false;
            state.vhConstituents[vh].push_back(&stripHit);
            break;
          }
        }
        if (!unique) {
          continue;
        }
        state.grid.addHit(&stripHit);
        recHitsSeen.insert(&stripHit);
      }
    }
  }

  // phase-1 rphi strip hits
  if (hasRphiStrips) {
    LogDebug("CosmicGridTrupletSeeder") << "have rphi hits ";
    for (auto ds : *rPhiStripHits) {
      for (const auto& stripHit : ds) {
        bool unique = true;
        for (const auto* vh : vhSeen) {
          if (vh->sharesInput(&stripHit, TrackingRecHit::some)) {
            unique = false;
            state.vhConstituents[vh].push_back(&stripHit);
            break;
          }
        }
        if (!unique) {
          continue;
        }
        state.grid.addHit(&stripHit);
        recHitsSeen.insert(&stripHit);
      }
    }
  }

  /// step 3: Add pixel hits if desired
  if (hasPixels) {
    LogDebug("CosmicGridTrupletSeeder") << "have Pixel hits ";
    for (auto ds : *pixelHitCollection) {
      for (const auto& pix : ds) {
        state.grid.addHit(&pix);
      }
    }
  }

  // now sort all bins of the grid by descending global y.
  state.grid.sort();
}

/// using the seeding grid, form triplets
void CosmicGridTripletSeeder::formTriplets(const CosmicGridTripletSeeder::TripletSeederEventState& state,
                                           std::vector<OrderedHitTriplet>& found) const {
  std::unordered_multiset<const BaseTrackerRecHit*> trackUsage;
  // loop downwards over the starting y bin, top to bottom
  for (int yBin = state.grid.nBinsY() - 1; yBin >= 0; --yBin) {
    // loop rectangularily over the x-z grid
    for (int xBin = 0; xBin < state.grid.nBinsX(); ++xBin) {
      for (int zBin = 0; zBin < state.grid.nBinsZ(); ++zBin) {
        formTriplets(xBin, yBin, zBin, state, found, trackUsage);
      }
    }
  }
  // sort by y-location of the lower hit, fall-back to upper if same
  std::sort(found.begin(), found.end(), [](const OrderedHitTriplet& t1, const OrderedHitTriplet& t2) {
    double ylow1 = t1.outer()->globalPosition().y();
    double ylow2 = t2.outer()->globalPosition().y();
    double ytop1 = t1.inner()->globalPosition().y();
    double ytop2 = t2.inner()->globalPosition().y();
    return (ylow1 == ylow2 ? ytop1 > ytop2 : ylow1 > ylow2);
  });
}

void CosmicGridTripletSeeder::formTriplets(int xBin,
                                           int yBin,
                                           int zBin,
                                           const TripletSeederEventState& state,
                                           std::vector<OrderedHitTriplet>& found,
                                           std::unordered_multiset<const BaseTrackerRecHit*>& trackUsage) const {
  const CartesianSeedingGrid& grid = state.grid;

  const auto& topHits = grid.getHits(xBin, yBin, zBin);
  // do not run triplet formation on empty cells.
  if (topHits.empty())
    return;

  std::vector<const BaseTrackerRecHit*> hitCands;
  hitCands.insert(hitCands.end(), topHits.begin(), topHits.end());

  if (yBin > 0) {
    for (int dxCenter = -1; dxCenter < 2; ++dxCenter) {
      if (xBin + dxCenter < 0 || xBin + dxCenter >= grid.nBinsX())
        continue;
      for (int dzCenter = -1; dzCenter < 2; ++dzCenter) {
        if (zBin + dzCenter < 0 || zBin + dzCenter >= grid.nBinsZ())
          continue;
        const auto& middleHits = grid.getHits(xBin + dxCenter, yBin - 1, zBin + dzCenter);
        hitCands.insert(hitCands.end(), middleHits.begin(), middleHits.end());
      }
    }
  }

  std::sort(hitCands.begin(), hitCands.end(), [](const BaseTrackerRecHit* h1, const BaseTrackerRecHit* h2) {
    return h1->globalPosition().y() > h2->globalPosition().y();
  });

  formTriplets(topHits, hitCands, hitCands, state, found, trackUsage);
}

/// get the triplets for one particular cell combination
void CosmicGridTripletSeeder::formTriplets(const std::vector<const BaseTrackerRecHit*>& topCands,
                                           const std::vector<const BaseTrackerRecHit*>& centerCands,
                                           const std::vector<const BaseTrackerRecHit*>& bottomCands,
                                           const TripletSeederEventState& state,
                                           std::vector<OrderedHitTriplet>& found,
                                           std::unordered_multiset<const BaseTrackerRecHit*>& trackUsage) const {
  for (const BaseTrackerRecHit* top : topCands) {
    if (trackUsage.count(top) > maxTripsPerSP_)
      continue;
    const auto& gTop = top->globalPosition();

    for (const BaseTrackerRecHit* center : centerCands) {
      if (trackUsage.count(center) > maxTripsPerSP_)
        continue;
      const auto& gCenter = center->globalPosition();

      if (gTop.y() < gCenter.y())
        continue;
      if (top->sameDetModule(*center))
        continue;

      for (const BaseTrackerRecHit* bottom : bottomCands) {
        if (trackUsage.count(bottom) > maxTripsPerSP_)
          continue;
        const auto& gBottom = bottom->globalPosition();

        if (top == bottom || center == bottom)
          continue;
        if (top->sameDetModule(*bottom) || center->sameDetModule(*bottom))
          continue;
        if (gCenter.y() < gBottom.y())
          continue;

        OrderedHitTriplet ps(top, center, bottom);
        if (!tripletOk(state, ps))
          continue;

        trackUsage.insert(top);
        trackUsage.insert(center);
        trackUsage.insert(bottom);
        found.push_back(ps);
      }
    }
  }
}
/// fit the triplets with kalman into trajectory seeds
void CosmicGridTripletSeeder::fitTriplets(const CosmicGridTripletSeeder::TripletSeederEventState& state,
                                          const std::vector<OrderedHitTriplet>& triplets,
                                          TrajectorySeedCollection& output) const {
  std::unordered_multiset<const TrackingRecHit*> seedPerSP;
  for (const OrderedHitTriplet& trip : triplets) {
    if (seedPerSP.count(trip.inner()) >= maxSeedsPerSP_ || seedPerSP.count(trip.middle()) >= maxSeedsPerSP_ ||
        seedPerSP.count(trip.outer()) >= maxSeedsPerSP_)
      continue;
    fitTriplet(state, trip, output, seedPerSP, true);
    if (tryBothDirections_)
      fitTriplet(state, trip, output, seedPerSP, false);
  }
}

/// fit of a single triplet into a trajectory seed
bool CosmicGridTripletSeeder::fitTriplet(const CosmicGridTripletSeeder::TripletSeederEventState& state,
                                         const OrderedHitTriplet& triplet,
                                         TrajectorySeedCollection& output,
                                         std::unordered_multiset<const TrackingRecHit*>& seedPerSP,
                                         bool goUp) const {
  typedef TrajectoryStateOnSurface TSOS;

  // arrange the triplet so that the hits appear in extrapolation order, with the *target hit* appearing first ("inner").
  // NB: They initially come in the order "top - middle - bottom" (in global y) when passed to this method.
  OrderedHitTriplet tripToUse = goUp ? triplet : OrderedHitTriplet{triplet.outer(), triplet.middle(), triplet.inner()};

  // top SP
  GlobalPoint target = state.tracker->idToDet((*(tripToUse.inner())).geographicalId())
                           ->surface()
                           .toGlobal((*(tripToUse.inner())).localPosition());

  // middle SP
  GlobalPoint middle = state.tracker->idToDet((*(tripToUse.middle())).geographicalId())
                           ->surface()
                           .toGlobal((*(tripToUse.middle())).localPosition());

  // bottom SP
  GlobalPoint start = state.tracker->idToDet((*(tripToUse.outer())).geographicalId())
                          ->surface()
                          .toGlobal((*(tripToUse.outer())).localPosition());

  // Initial momentum estimate at the start point
  // using the fast helix helper.
  // Note: arg order of FastHelix is "outer middle vertex",
  // so we will get a momentum pointing *up* when targeting the top SP,
  // and pointing down when targeting the bottom SP.
  FastHelix helix(target, middle, start, state.magfield->nominalValue(), state.magfield);

  // extract momentum and charge - flip when needed to ensure a momentum
  // pointing down (cosmic trajectory)
  GlobalVector gv = (goUp ? -1 : 1) * helix.stateAtVertex().momentum();
  float ch = (goUp ? -1 : 1) * helix.stateAtVertex().charge();

  // initial cleaning.
  float Mom = gv.mag();
  if (Mom > 1000000 || edm::isNotFinite(Mom)) {
    return false;
  }
  if (gv.perp() < 0.5 * ptMin_) {
    return false;
  }

  // Next use a KF to refine the parameter estimate.
  // Propagate from the start to the target, and track
  // relation to the momentum direction.
  const Propagator* propagator = goUp ? state.thePropagatorOp.get() : state.thePropagatorAl.get();

  edm::OwnVector<TrackingRecHit> hits;
  std::vector<const BaseTrackerRecHit*> seedHits;

  // build our hit array in the order pointing towards the target ("inner")
  for (const BaseTrackerRecHit* hit : {tripToUse.outer(), tripToUse.middle(), tripToUse.inner()}) {
    const VectorHit* vh = dynamic_cast<const VectorHit*>(hit);
    if (vh) {
      auto found = state.vhConstituents.find(vh);
      if (found != state.vhConstituents.end()) {
        for (auto& component : found->second) {
          seedHits.push_back(component);
        }
      } else {
        edm::LogWarning("CosmicGridTripletSeeder")
            << "Did not find any constitutents for a vector hit - are our inputs consistent?";
        seedHits.push_back(hit);
      }
    } else {
      seedHits.push_back(hit);
    }
  }
  // re-sort, to account for the vector hit components being possibly out-of-order
  std::sort(seedHits.begin(), seedHits.end(), [&](const BaseTrackerRecHit* h1, const BaseTrackerRecHit* h2) {
    return goUp ? h1->globalPosition().y() < h2->globalPosition().y()
                : h1->globalPosition().y() > h2->globalPosition().y();
  });

  // starting parameters: Lowest point, estimated momentum from fast helix.
  GlobalPoint updatedStartPos = seedHits.front()->globalPosition();
  GlobalTrajectoryParameters Gtp(updatedStartPos, gv, int(ch), state.magfield);
  FreeTrajectoryState CosmicSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
  CosmicSeed.rescaleError(100);
  TSOS propagated, updated;

  // now incrementally propagate towards the target point
  for (size_t ih = 0; ih < seedHits.size(); ++ih) {
    if (ih == 0) {
      propagated =
          propagator->propagate(CosmicSeed, state.tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
    } else {
      propagated = propagator->propagate(updated, state.tracker->idToDet((*seedHits[ih]).geographicalId())->surface());
    }
    if (!propagated.isValid()) {
      return false;
    }

    // clone the hit based on the propagated state
    SeedingHitSet::ConstRecHitPointer tthp = seedHits[ih];
    auto newtth = static_cast<SeedingHitSet::RecHitPointer>(state.cloner(*tthp, propagated));
    updated = state.theUpdator->update(propagated, *newtth);
    hits.push_back(newtth);

    if (!updated.isValid()) {
      return false;
    }
  }

  // some more cleaning
  if (updated.globalMomentum().perp() < ptMin_) {
    return false;
  }
  if (updated.globalMomentum().mag() < ptMin_) {
    return false;
  }

  PTrajectoryStateOnDet const& PTraj =
      trajectoryStateTransform::persistentState(updated, hits.back().geographicalId().rawId());

  // and build the final seed.
  output.push_back(TrajectorySeed(PTraj, hits, goUp ? oppositeToMomentum : alongMomentum));

  if (output.size() > size_t(500)) {
    output.clear();
    edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
    return false;
  }
  seedPerSP.insert(triplet.inner());
  seedPerSP.insert(triplet.middle());
  seedPerSP.insert(triplet.outer());

  return true;
}
