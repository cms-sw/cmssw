#include "RecoTracker/SpecialSeedGenerators/interface/SeedGeneratorForCRack.h"
#include "RecoTracker/TkHitPairs/interface/CosmicLayerPairs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
void SeedGeneratorForCRack::init(const SiStripRecHit2DCollection& collstereo,
                                 const SiStripRecHit2DCollection& collrphi,
                                 const SiStripMatchedRecHit2DCollection& collmatched,
                                 const edm::EventSetup& iSetup) {
  magfield = iSetup.getHandle(theMagfieldToken);
  tracker = iSetup.getHandle(theTrackerToken);
  thePropagatorAl = new PropagatorWithMaterial(alongMomentum, 0.1057, &(*magfield));
  thePropagatorOp = new PropagatorWithMaterial(oppositeToMomentum, 0.1057, &(*magfield));
  theUpdator = new KFUpdator();

  // get the transient builder
  TTTRHBuilder = &iSetup.getData(theTTRHToken);

  GeometricSearchTracker const& track = iSetup.getData(theSearchTrackerToken);
  TrackerTopology const& ttopo = iSetup.getData(theTTopoToken);

  CosmicLayerPairs cosmiclayers(geometry, collrphi, collmatched, track, ttopo);
  thePairGenerator = new CosmicHitPairGenerator(cosmiclayers, *tracker);
  HitPairs.clear();
  thePairGenerator->hitPairs(region, HitPairs);
  LogDebug("CosmicSeedFinder") << "Initialized with " << HitPairs.size() << " hit pairs" << std::endl;
}

SeedGeneratorForCRack::SeedGeneratorForCRack(edm::ParameterSet const& conf, edm::ConsumesCollector iCC)
    : theMagfieldToken(iCC.esConsumes()),
      theTrackerToken(iCC.esConsumes()),
      theSearchTrackerToken(iCC.esConsumes()),
      theTTopoToken(iCC.esConsumes()),
      theTTRHToken(iCC.esConsumes(edm::ESInputTag("", conf.getParameter<std::string>("TTRHBuilder")))),
      region(conf.getParameter<double>("ptMin"),
             conf.getParameter<double>("originRadius"),
             conf.getParameter<double>("originHalfLength"),
             conf.getParameter<double>("originZPosition")) {
  seedpt = conf.getParameter<double>("SeedPt");
  geometry = conf.getUntrackedParameter<std::string>("GeometricStructure", "STANDARD");
  multipleScatteringFactor = conf.getUntrackedParameter<double>("multipleScatteringFactor", 1.0);
  seedMomentum = conf.getUntrackedParameter<double>("SeedMomentum", 1);
}

void SeedGeneratorForCRack::run(TrajectorySeedCollection& output, const edm::EventSetup& iSetup) {
  seeds(output, iSetup, region);
  delete thePairGenerator;
  delete thePropagatorAl;
  delete thePropagatorOp;
  delete theUpdator;
}

void SeedGeneratorForCRack::seeds(TrajectorySeedCollection& output,
                                  const edm::EventSetup& iSetup,
                                  const TrackingRegion& region) {
  for (unsigned int is = 0; is < HitPairs.size(); is++) {
    GlobalPoint inner = tracker->idToDet((*(HitPairs[is].inner())).geographicalId())
                            ->surface()
                            .toGlobal((*(HitPairs[is].inner())).localPosition());
    GlobalPoint outer = tracker->idToDet((*(HitPairs[is].outer())).geographicalId())
                            ->surface()
                            .toGlobal((*(HitPairs[is].outer())).localPosition());

    LogDebug("CosmicSeedFinder") << "inner point of the seed " << inner << " outer point of the seed " << outer;
    SeedingHitSet::ConstRecHitPointer inrhit = HitPairs[is].inner();
    SeedingHitSet::ConstRecHitPointer outrhit = HitPairs[is].outer();

    edm::OwnVector<TrackingRecHit> hits;
    hits.push_back(HitPairs[is].outer()->hit()->clone());

    for (int i = 0; i < 2; i++) {
      //FIRST STATE IS CALCULATED CONSIDERING THAT THE CHARGE CAN BE POSITIVE OR NEGATIVE
      int predsign = (2 * i) - 1;
      if ((outer.y() - inner.y()) > 0) {
        GlobalVector momentum = GlobalVector(inner - outer);
        momentum = momentum.unit() * seedMomentum;
        GlobalTrajectoryParameters Gtp(inner, momentum, predsign, &(*magfield));
        AlgebraicSymMatrix55 errMatrix = ROOT::Math::SMatrixIdentity();
        TSOS innerState = TSOS(Gtp,
                               CurvilinearTrajectoryError(errMatrix),
                               tracker->idToDet((HitPairs[is].inner()->hit())->geographicalId())->surface());
        const TSOS innerUpdated = theUpdator->update(innerState, *inrhit);
        //Cosmic Seed update inner...
        LogDebug("CosmicSeedFinder") << " FirstTSOS " << innerUpdated;

        //First propagation
        const TSOS outerState = thePropagatorOp->propagate(
            innerUpdated, tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface());
        if (outerState.isValid()) {
          LogDebug("CosmicSeedFinder") << "outerState " << outerState;
          TSOS outerUpdated = theUpdator->update(outerState, *outrhit);
          //fudge factor for multiple scattering
          outerUpdated.rescaleError(multipleScatteringFactor);
          if (outerUpdated.isValid()) {
            LogDebug("CosmicSeedFinder") << "outerUpdated " << outerUpdated;

            PTrajectoryStateOnDet PTraj = trajectoryStateTransform::persistentState(
                outerUpdated, (*(HitPairs[is].outer())).geographicalId().rawId());
            output.push_back(TrajectorySeed(PTraj, hits, alongMomentum));

          } else
            edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
        } else
          edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";

      } else {
        GlobalVector momentum = GlobalVector(outer - inner);
        momentum = momentum.unit() * seedMomentum;
        GlobalTrajectoryParameters Gtp(inner, momentum, predsign, &(*magfield));
        AlgebraicSymMatrix55 errMatrix = ROOT::Math::SMatrixIdentity();
        TSOS innerState = TSOS(Gtp,
                               CurvilinearTrajectoryError(errMatrix),
                               tracker->idToDet((HitPairs[is].inner()->hit())->geographicalId())->surface());
        const TSOS innerUpdated = theUpdator->update(innerState, *inrhit);
        LogDebug("CosmicSeedFinder") << " FirstTSOS " << innerState;

        //First propagation
        const TSOS outerState = thePropagatorOp->propagate(
            innerUpdated, tracker->idToDet((*(HitPairs[is].outer())).geographicalId())->surface());
        if (outerState.isValid()) {
          LogDebug("CosmicSeedFinder") << "outerState " << outerState;
          TSOS outerUpdated = theUpdator->update(outerState, *outrhit);
          //fudge factor for multiple scattering
          outerUpdated.rescaleError(multipleScatteringFactor);
          if (outerUpdated.isValid()) {
            LogDebug("CosmicSeedFinder") << "outerUpdated " << outerUpdated;
            PTrajectoryStateOnDet PTraj = trajectoryStateTransform::persistentState(
                outerUpdated, (*(HitPairs[is].outer())).geographicalId().rawId());
            output.push_back(TrajectorySeed(PTraj, hits, oppositeToMomentum));

          } else
            edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first update failed ";
        } else
          edm::LogWarning("CosmicSeedFinder") << " SeedForCosmics first propagation failed ";
      }
    }
  }
}
