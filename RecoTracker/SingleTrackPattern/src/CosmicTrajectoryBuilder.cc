//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>

#include "RecoTracker/SingleTrackPattern/interface/CosmicTrajectoryBuilder.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateWithArbitraryError.h"
using namespace std;

namespace {

  /// @brief check for hits already on a seed. As the latter are cloned,
  /// use identical global position (with a tolerance) rather than
  /// testing pointer equality.
  /// @param globHit: global-frame position of the hit to check against the seed
  /// @param seedHits: List of hits on the seed
  /// @return true if the passed position matches one of the hits on the seed
  ///         within the given tolerance.
  bool onSeed(const GlobalPoint &globHit, const std::vector<std::shared_ptr<const TrackingRecHit>> &seedHits) {
    /// tolerance: 1 um - should be well below expected distance between genuine separate hits
    static constexpr double epsilon = 0.0001 * 0.0001;
    for (const std::shared_ptr<const TrackingRecHit> &rh : seedHits) {
      if ((rh->globalPosition() - globHit).mag2() < epsilon) {
        return true;
      }
    }
    return false;
  };
}  // namespace

CosmicTrajectoryBuilder::CosmicTrajectoryBuilder(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : magfieldToken_(iC.esConsumes()),
      trackerToken_(iC.esConsumes()),
      builderToken_(iC.esConsumes(edm::ESInputTag("", conf.getParameter<std::string>("TTRHBuilder")))) {
  //minimum number of hits per tracks

  theMinHits = conf.getParameter<int>("MinHits");
  //cut on chi2
  chi2cut = conf.getParameter<double>("Chi2Cut");
  edm::LogInfo("CosmicTrackFinder") << "Minimum number of hits " << theMinHits << " Cut on Chi2= " << chi2cut;

  geometry = conf.getUntrackedParameter<std::string>("GeometricStructure", "STANDARD");
}

CosmicTrajectoryBuilder::~CosmicTrajectoryBuilder() {}

void CosmicTrajectoryBuilder::init(const edm::EventSetup &es, bool seedplus) {
  // FIXME: this is a memory leak generator

  //services
  magfield = &es.getData(magfieldToken_);
  tracker = &es.getData(trackerToken_);

  if (seedplus) {
    seed_plus = true;
    thePropagator = new PropagatorWithMaterial(alongMomentum, 0.1057, &(*magfield));
    thePropagatorOp = new PropagatorWithMaterial(oppositeToMomentum, 0.1057, &(*magfield));
  } else {
    seed_plus = false;
    thePropagator = new PropagatorWithMaterial(oppositeToMomentum, 0.1057, &(*magfield));
    thePropagatorOp = new PropagatorWithMaterial(alongMomentum, 0.1057, &(*magfield));
  }

  theUpdator = new KFUpdator();
  theEstimator = new Chi2MeasurementEstimator(chi2cut);

  RHBuilder = &es.getData(builderToken_);
  hitCloner = static_cast<TkTransientTrackingRecHitBuilder const *>(RHBuilder)->cloner();

  theFitter = new KFTrajectoryFitter(*thePropagator, *theUpdator, *theEstimator);
  theFitter->setHitCloner(&hitCloner);

  theSmoother = new KFTrajectorySmoother(*thePropagatorOp, *theUpdator, *theEstimator);
  theSmoother->setHitCloner(&hitCloner);
}

void CosmicTrajectoryBuilder::run(
    const TrajectorySeedCollection &collseed,
    std::optional<std::reference_wrapper<const SiStripRecHit2DCollection>> collstereo,
    std::optional<std::reference_wrapper<const SiStripRecHit2DCollection>> collrphi,
    std::optional<std::reference_wrapper<const SiStripMatchedRecHit2DCollection>> collmatched,
    std::optional<std::reference_wrapper<const SiPixelRecHitCollection>> collpixel,
    std::optional<std::reference_wrapper<const Phase2TrackerRecHit1DCollectionNew>> p2RecHits1D,
    const edm::EventSetup &es,
    edm::Event &e,
    vector<Trajectory> &trajoutput) {
  std::vector<Trajectory> trajSmooth;
  std::vector<Trajectory>::iterator trajIter;

  TrajectorySeedCollection::const_iterator iseed;
  unsigned int IS = 0;
  for (iseed = collseed.begin(); iseed != collseed.end(); iseed++) {
    bool seedplus = ((*iseed).direction() == alongMomentum);
    init(es, seedplus);
    hits.clear();
    trajFit.clear();
    trajSmooth.clear();
    //order all the hits
    vector<const TrackingRecHit *> allHits =
        SortHits(collstereo, collrphi, collmatched, collpixel, p2RecHits1D, *iseed);
    Trajectory startingTraj = createStartingTrajectory(*iseed);
    AddHit(startingTraj, allHits);
    for (trajIter = trajFit.begin(); trajIter != trajFit.end(); trajIter++) {
      trajSmooth = theSmoother->trajectories((*trajIter));
    }
    for (trajIter = trajSmooth.begin(); trajIter != trajSmooth.end(); trajIter++) {
      if ((*trajIter).isValid()) {
        trajoutput.push_back((*trajIter));
      }
    }
    delete theUpdator;
    delete theEstimator;
    delete thePropagator;
    delete thePropagatorOp;
    delete theFitter;
    delete theSmoother;
    //Only the first 30 seeds are considered
    if (IS > 30)
      return;
    IS++;
  }
}

Trajectory CosmicTrajectoryBuilder::createStartingTrajectory(const TrajectorySeed &seed) const {
  Trajectory result(seed, seed.direction());
  std::vector<TM> &&seedMeas = seedMeasurements(seed);
  for (auto &i : seedMeas)
    result.push(std::move(i));
  return result;
}

std::vector<TrajectoryMeasurement> CosmicTrajectoryBuilder::seedMeasurements(const TrajectorySeed &seed) const {
  std::vector<TrajectoryMeasurement> result;
  auto const &hitRange = seed.recHits();
  for (auto ihit = hitRange.begin(); ihit != hitRange.end(); ihit++) {
    //RC TransientTrackingRecHit* recHit = RHBuilder->build(&(*ihit));
    TransientTrackingRecHit::RecHitPointer recHit = RHBuilder->build(&(*ihit));
    const GeomDet *hitGeomDet = (&(*tracker))->idToDet(ihit->geographicalId());
    TSOS invalidState(new BasicSingleTrajectoryState(hitGeomDet->surface()));

    if (ihit == hitRange.end() - 1) {
      TSOS updatedState = startingTSOS(seed);
      result.emplace_back(invalidState, updatedState, recHit);
    } else {
      result.emplace_back(invalidState, recHit);
    }
  }

  return result;
}

vector<const TrackingRecHit *> CosmicTrajectoryBuilder::SortHits(
    std::optional<std::reference_wrapper<const SiStripRecHit2DCollection>> collstereo,
    std::optional<std::reference_wrapper<const SiStripRecHit2DCollection>> collrphi,
    std::optional<std::reference_wrapper<const SiStripMatchedRecHit2DCollection>> collmatched,
    std::optional<std::reference_wrapper<const SiPixelRecHitCollection>> collpixel,
    std::optional<std::reference_wrapper<const Phase2TrackerRecHit1DCollectionNew>> p2RecHits1D,
    const TrajectorySeed &seed) {
  //The Hits corresponding to the seed are discarded
  //At the end all the hits are sorted in y
  vector<const TrackingRecHit *> allHits;

  for (auto const &recHit : seed.recHits()) {
    hits.push_back((RHBuilder->build(&recHit)));
  }

  if (collpixel.has_value()) {
    for (const SiPixelRecHit &iPix : collpixel.value().get().data()) {
      if (onSeed(RHBuilder->build(&iPix)->globalPosition(), hits)) {
        continue;
      }
      allHits.push_back(&iPix);
    }
  }

  if (collrphi.has_value()) {
    for (const SiStripRecHit2D &iStrip : collrphi.value().get().data()) {
      if (onSeed(RHBuilder->build(&iStrip)->globalPosition(), hits)) {
        continue;
      }
      allHits.push_back(&iStrip);
    }
  }
  if (collstereo.has_value()) {
    for (const SiStripRecHit2D &istrip : collstereo.value().get().data()) {
      if (onSeed(RHBuilder->build(&istrip)->globalPosition(), hits)) {
        continue;
      }
      allHits.push_back(&istrip);
    }
  }

  if (p2RecHits1D.has_value()) {
    for (const Phase2TrackerRecHit1D &otHit : p2RecHits1D.value().get().data()) {
      if (onSeed(RHBuilder->build(&otHit)->globalPosition(), hits)) {
        continue;
      }
      allHits.push_back(&otHit);
    }
  }
  if (seed_plus)
    stable_sort(allHits.begin(), allHits.end(), CompareHitY_plus(*tracker));
  else
    stable_sort(allHits.begin(), allHits.end(), CompareHitY(*tracker));

  return allHits;
}

TrajectoryStateOnSurface CosmicTrajectoryBuilder::startingTSOS(const TrajectorySeed &seed) const {
  PTrajectoryStateOnDet pState(seed.startingState());
  const GeomDet *gdet = (&(*tracker))->idToDet(DetId(pState.detId()));
  TSOS State = trajectoryStateTransform::transientState(pState, &(gdet->surface()), &(*magfield));
  return State;
}

void CosmicTrajectoryBuilder::AddHit(Trajectory &traj, const vector<const TrackingRecHit *> &Hits) {
  unsigned int icosm2;
  unsigned int ibestdet;
  float chi2min;
  for (unsigned int icosmhit = 0; icosmhit < Hits.size(); icosmhit++) {
    GlobalPoint gphit = RHBuilder->build(Hits[icosmhit])->globalPosition();
    unsigned int iraw = Hits[icosmhit]->geographicalId().rawId();
    LogDebug("CosmicTrackFinder") << " HIT POSITION " << gphit;
    //RC TransientTrackingRecHit* tmphit=RHBuilder->build(Hits[icosmhit]);
    TransientTrackingRecHit::RecHitPointer tmphit = RHBuilder->build(Hits[icosmhit]);
    TSOS prSt = thePropagator->propagate(traj.lastMeasurement().updatedState(),
                                         tracker->idToDet(Hits[icosmhit]->geographicalId())->surface());

    //After propagating the trajectory to a detector,
    //find the most compatible hit in the det
    chi2min = 20000000;
    ibestdet = 1000;
    if (prSt.isValid()) {
      LocalPoint prLoc = prSt.localPosition();
      LogDebug("CosmicTrackFinder") << "STATE PROPAGATED AT DET " << iraw << " " << prSt;
      for (icosm2 = icosmhit; icosm2 < Hits.size(); icosm2++) {
        if (iraw == Hits[icosm2]->geographicalId().rawId()) {
          TransientTrackingRecHit::RecHitPointer tmphit = RHBuilder->build(Hits[icosm2]);
          float contr = theEstimator->estimate(prSt, *tmphit).second;
          if (contr < chi2min) {
            chi2min = contr;
            ibestdet = icosm2;
          }
          if (icosm2 != icosmhit)
            icosmhit++;

        } else
          icosm2 = Hits.size();
      }

      if (chi2min < chi2cut)
        LogDebug("CosmicTrackFinder") << "Chi2 contribution for hit at "
                                      << RHBuilder->build(Hits[ibestdet])->globalPosition() << " is " << chi2min;

      if (traj.foundHits() < 3 && (chi2min < chi2cut)) {
        //check on the first hit after the seed
        GlobalVector ck = RHBuilder->build(Hits[ibestdet])->globalPosition() -
                          traj.firstMeasurement().updatedState().globalPosition();
        if ((abs(ck.x() / ck.y()) > 2) || (abs(ck.z() / ck.y()) > 2))
          chi2min = 300;
      }
      if (chi2min < chi2cut) {
        if (abs(prLoc.x()) < 25 && abs(prLoc.y()) < 25) {
          TransientTrackingRecHit::RecHitPointer tmphitbestdet = RHBuilder->build(Hits[ibestdet]);
          TSOS UpdatedState = theUpdator->update(prSt, *tmphitbestdet);
          if (UpdatedState.isValid()) {
            traj.push(TM(prSt, UpdatedState, RHBuilder->build(Hits[ibestdet]), chi2min));
            LogDebug("CosmicTrackFinder") << "STATE UPDATED WITH HIT AT POSITION " << tmphitbestdet->globalPosition()
                                          << UpdatedState << " " << traj.chiSquared();

            hits.push_back(tmphitbestdet);
          }
        } else
          LogDebug("CosmicTrackFinder") << " Hits outside module surface " << prLoc;
      } else
        LogDebug("CosmicTrackFinder") << " State can not be updated with hit at position " << gphit;
    } else
      LogDebug("CosmicTrackFinder") << " State can not be propagated at det " << iraw;
  }

  if (qualityFilter(traj)) {
    const TrajectorySeed &tmpseed = traj.seed();
    if (thePropagatorOp
            ->propagate(traj.lastMeasurement().updatedState(),
                        tracker->idToDet((*hits.begin())->geographicalId())->surface())
            .isValid()) {
      TSOS startingState = thePropagatorOp->propagate(traj.lastMeasurement().updatedState(),
                                                      tracker->idToDet((*hits.begin())->geographicalId())->surface());

      trajFit = theFitter->fit(tmpseed, hits, startingState);
    }
  }
}

bool CosmicTrajectoryBuilder::qualityFilter(const Trajectory &traj) {
  int ngoodhits = 0;
  if (geometry == "MTCC") {
    auto hits = traj.recHits();
    for (auto hit = hits.begin(); hit != hits.end(); hit++) {
      unsigned int iid = (*hit)->hit()->geographicalId().rawId();
      //CHECK FOR 3 hits r-phi
      if (((iid >> 0) & 0x3) != 1)
        ngoodhits++;
    }
  } else
    ngoodhits = traj.foundHits();

  if (ngoodhits >= theMinHits) {
    return true;
  } else {
    return false;
  }
}
