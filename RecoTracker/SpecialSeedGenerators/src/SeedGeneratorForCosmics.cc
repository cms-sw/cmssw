#include "RecoTracker/SpecialSeedGenerators/interface/SeedGeneratorForCosmics.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "RecoTracker/PixelSeeding/interface/CosmicLayerTriplets.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
void SeedGeneratorForCosmics::init(const SiStripRecHit2DCollection& collstereo,
                                   const SiStripRecHit2DCollection& collrphi,
                                   const SiStripMatchedRecHit2DCollection& collmatched,
                                   const edm::EventSetup& iSetup) {
  magfield = iSetup.getHandle(theMagfieldToken);
  tracker = iSetup.getHandle(theTrackerToken);
  thePropagatorAl = new PropagatorWithMaterial(alongMomentum, 0.1057, &(*magfield));
  thePropagatorOp = new PropagatorWithMaterial(oppositeToMomentum, 0.1057, &(*magfield));
  theUpdator = new KFUpdator();

  LogDebug("CosmicSeedFinder") << " Hits built with  " << hitsforseeds << " hits";

  GeometricSearchTracker const& track = iSetup.getData(theSearchTrackerToken);
  TrackerTopology const& ttopo = iSetup.getData(theTTopoToken);

  CosmicLayerPairs cosmiclayers(geometry, collrphi, collmatched, track, ttopo);

  thePairGenerator = new CosmicHitPairGenerator(cosmiclayers, *tracker);
  HitPairs.clear();
  if ((hitsforseeds == "pairs") || (hitsforseeds == "pairsandtriplets")) {
    thePairGenerator->hitPairs(region, HitPairs);
  }

  CosmicLayerTriplets cosmiclayers2(geometry, collrphi, track, ttopo);
  theTripletGenerator = new CosmicHitTripletGenerator(cosmiclayers2, *tracker);
  HitTriplets.clear();
  if ((hitsforseeds == "triplets") || (hitsforseeds == "pairsandtriplets")) {
    theTripletGenerator->hitTriplets(region, HitTriplets);
  }
}

SeedGeneratorForCosmics::SeedGeneratorForCosmics(edm::ParameterSet const& conf, edm::ConsumesCollector iCC)
    : maxSeeds_(conf.getParameter<int32_t>("maxSeeds")),
      theMagfieldToken(iCC.esConsumes()),
      theTrackerToken(iCC.esConsumes()),
      theSearchTrackerToken(iCC.esConsumes()),
      theTTopoToken(iCC.esConsumes()) {
  float ptmin = conf.getParameter<double>("ptMin");
  float originradius = conf.getParameter<double>("originRadius");
  float halflength = conf.getParameter<double>("originHalfLength");
  float originz = conf.getParameter<double>("originZPosition");
  seedpt = conf.getParameter<double>("SeedPt");

  geometry = conf.getUntrackedParameter<std::string>("GeometricStructure", "STANDARD");
  region = GlobalTrackingRegion(ptmin, originradius, halflength, originz);
  hitsforseeds = conf.getUntrackedParameter<std::string>("HitsForSeeds", "pairs");
  edm::LogInfo("SeedGeneratorForCosmics")
      << " PtMin of track is " << ptmin << " The Radius of the cylinder for seeds is " << originradius << "cm"
      << " The set Seed Momentum" << seedpt;

  //***top-bottom
  positiveYOnly = conf.getParameter<bool>("PositiveYOnly");
  negativeYOnly = conf.getParameter<bool>("NegativeYOnly");
  //***
}

void SeedGeneratorForCosmics::run(const SiStripRecHit2DCollection& collstereo,
                                  const SiStripRecHit2DCollection& collrphi,
                                  const SiStripMatchedRecHit2DCollection& collmatched,
                                  const edm::EventSetup& c,
                                  TrajectorySeedCollection& output) {
  init(collstereo, collrphi, collmatched, c);
  seeds(output, region);
  delete thePairGenerator;
  delete theTripletGenerator;
  delete thePropagatorAl;
  delete thePropagatorOp;
  delete theUpdator;
}
bool SeedGeneratorForCosmics::seeds(TrajectorySeedCollection& output, const TrackingRegion& region) {
  LogDebug("CosmicSeedFinder") << "Number of triplets " << HitTriplets.size();
  LogDebug("CosmicSeedFinder") << "Number of pairs " << HitPairs.size();

  for (unsigned int it = 0; it < HitTriplets.size(); it++) {
    //const TrackingRecHit *hit = &(
    //    const TrackingRecHit* hit = it->hits();

    //   GlobalPoint inner = tracker->idToDet(HitTriplets[it].inner().RecHit()->
    // 					 geographicalId())->surface().
    //       toGlobal(HitTriplets[it].inner().RecHit()->localPosition());
    //const TrackingRecHit  *innerhit =  &(*HitTriplets[it].inner());
    //const TrackingRecHit  *middlehit =  &(*HitTriplets[it].middle());

    GlobalPoint inner = tracker->idToDet((*(HitTriplets[it].inner())).geographicalId())
                            ->surface()
                            .toGlobal((*(HitTriplets[it].inner())).localPosition());

    GlobalPoint middle = tracker->idToDet((*(HitTriplets[it].middle())).geographicalId())
                             ->surface()
                             .toGlobal((*(HitTriplets[it].middle())).localPosition());

    GlobalPoint outer = tracker->idToDet((*(HitTriplets[it].outer())).geographicalId())
                            ->surface()
                            .toGlobal((*(HitTriplets[it].outer())).localPosition());

    SeedingHitSet::ConstRecHitPointer outrhit = HitTriplets[it].outer();
    //***top-bottom
    SeedingHitSet::ConstRecHitPointer innrhit = HitTriplets[it].inner();
    if (positiveYOnly && (outrhit->globalPosition().y() < 0 || innrhit->globalPosition().y() < 0 ||
                          outrhit->globalPosition().y() < innrhit->globalPosition().y()))
      continue;
    if (negativeYOnly && (outrhit->globalPosition().y() > 0 || innrhit->globalPosition().y() > 0 ||
                          outrhit->globalPosition().y() > innrhit->globalPosition().y()))
      continue;
    //***

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitTriplets[it].outer()->hit()->clone());
    FastHelix helix(inner, middle, outer, magfield->nominalValue(), &(*magfield));
    GlobalVector gv = helix.stateAtVertex().momentum();
    float ch = helix.stateAtVertex().charge();
    float mom2 = gv.x() * gv.x() + gv.y() * gv.y() + gv.z() * gv.z();
    if (mom2 > 1e06f || edm::isNotFinite(mom2))
      continue;  // ChangedByDaniele

    if (gv.y() > 0) {
      gv = -1. * gv;
      ch = -1. * ch;
    }

    GlobalTrajectoryParameters Gtp(outer, gv, int(ch), &(*magfield));
    FreeTrajectoryState CosmicSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
    if ((outer.y() - inner.y()) > 0) {
      const TSOS outerState = thePropagatorAl->propagate(
          CosmicSeed, tracker->idToDet((*(HitTriplets[it].outer())).geographicalId())->surface());
      if (outerState.isValid()) {
        LogDebug("CosmicSeedFinder") << "outerState " << outerState;
        const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);
        if (outerUpdated.isValid()) {
          LogDebug("CosmicSeedFinder") << "outerUpdated " << outerUpdated;

          output.push_back(TrajectorySeed(trajectoryStateTransform::persistentState(
                                              outerUpdated, (*(HitTriplets[it].outer())).geographicalId().rawId()),
                                          hits,
                                          alongMomentum));

          if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
            edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
            output.clear();
            return false;
          }
        }
      }
    } else {
      const TSOS outerState = thePropagatorOp->propagate(
          CosmicSeed, tracker->idToDet((*(HitTriplets[it].outer())).geographicalId())->surface());
      if (outerState.isValid()) {
        LogDebug("CosmicSeedFinder") << "outerState " << outerState;
        const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);
        if (outerUpdated.isValid()) {
          LogDebug("CosmicSeedFinder") << "outerUpdated " << outerUpdated;

          output.push_back(TrajectorySeed(trajectoryStateTransform::persistentState(
                                              outerUpdated, (*(HitTriplets[it].outer())).geographicalId().rawId()),
                                          hits,
                                          oppositeToMomentum));

          if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
            edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
            output.clear();
            return false;
          }
        }
      }
    }
  }

  for (unsigned int is = 0; is < HitPairs.size(); is++) {
    GlobalPoint inner = tracker->idToDet((*(HitPairs[is].inner())).geographicalId())
                            ->surface()
                            .toGlobal((*(HitPairs[is].inner())).localPosition());
    GlobalPoint outer = tracker->idToDet((*(HitPairs[is].outer())).geographicalId())
                            ->surface()
                            .toGlobal((*(HitPairs[is].outer())).localPosition());

    LogDebug("CosmicSeedFinder") << "inner point of the seed " << inner << " outer point of the seed " << outer;
    SeedingHitSet::ConstRecHitPointer outrhit = HitPairs[is].outer();
    //***top-bottom
    SeedingHitSet::ConstRecHitPointer innrhit = HitPairs[is].inner();
    if (positiveYOnly && (outrhit->globalPosition().y() < 0 || innrhit->globalPosition().y() < 0 ||
                          outrhit->globalPosition().y() < innrhit->globalPosition().y()))
      continue;
    if (negativeYOnly && (outrhit->globalPosition().y() > 0 || innrhit->globalPosition().y() > 0 ||
                          outrhit->globalPosition().y() > innrhit->globalPosition().y()))
      continue;
    //***

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[is].outer()->hit()->clone());
    //    hits.push_back(HitPairs[is].inner()->clone());

    for (int i = 0; i < 2; i++) {
      //FIRST STATE IS CALCULATED CONSIDERING THAT THE CHARGE CAN BE POSITIVE OR NEGATIVE
      int predsign = (2 * i) - 1;
      if ((outer.y() - inner.y()) > 0) {
        GlobalTrajectoryParameters Gtp(
            outer, (inner - outer) * (seedpt / (inner - outer).mag()), predsign, &(*magfield));

        FreeTrajectoryState CosmicSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));

        LogDebug("CosmicSeedFinder") << " FirstTSOS " << CosmicSeed;
        //First propagation
        const TSOS outerState = thePropagatorAl->propagate(
            CosmicSeed, tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface());
        if (outerState.isValid()) {
          LogDebug("CosmicSeedFinder") << "outerState " << outerState;
          const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);
          if (outerUpdated.isValid()) {
            LogDebug("CosmicSeedFinder") << "outerUpdated " << outerUpdated;

            PTrajectoryStateOnDet const& PTraj = trajectoryStateTransform::persistentState(
                outerUpdated, (*(HitPairs[is].outer())).geographicalId().rawId());

            output.push_back(TrajectorySeed(PTraj, hits, alongMomentum));

            if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
              edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
              output.clear();
              return false;
            }

          } else
            edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
        } else
          edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";

      } else {
        GlobalTrajectoryParameters Gtp(
            outer, (outer - inner) * (seedpt / (outer - inner).mag()), predsign, &(*magfield));
        FreeTrajectoryState CosmicSeed(Gtp, CurvilinearTrajectoryError(AlgebraicSymMatrix55(AlgebraicMatrixID())));
        LogDebug("CosmicSeedFinder") << " FirstTSOS " << CosmicSeed;
        //First propagation
        const TSOS outerState = thePropagatorAl->propagate(
            CosmicSeed, tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface());
        if (outerState.isValid()) {
          LogDebug("CosmicSeedFinder") << "outerState " << outerState;
          const TSOS outerUpdated = theUpdator->update(outerState, *outrhit);
          if (outerUpdated.isValid()) {
            LogDebug("CosmicSeedFinder") << "outerUpdated " << outerUpdated;

            PTrajectoryStateOnDet const& PTraj = trajectoryStateTransform::persistentState(
                outerUpdated, (*(HitPairs[is].outer())).geographicalId().rawId());

            output.push_back(TrajectorySeed(PTraj, hits, oppositeToMomentum));

            if ((maxSeeds_ > 0) && (output.size() > size_t(maxSeeds_))) {
              edm::LogError("TooManySeeds") << "Found too many seeds, bailing out.\n";
              output.clear();
              return false;
            }

          } else
            edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
        } else
          edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";
      }
    }
  }
  return true;
}
