#ifndef RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreator_H
#define RecoTracker_TkSeedGenerator_SeedFromConsecutiveHitsCreator_H
#include "FWCore/Utilities/interface/Visibility.h"

#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

class FreeTrajectoryState;

class dso_hidden SeedFromConsecutiveHitsCreator : public SeedCreator {
public:
  SeedFromConsecutiveHitsCreator(const edm::ParameterSet &, edm::ConsumesCollector &&);

  ~SeedFromConsecutiveHitsCreator() override;

  static void fillDescriptions(edm::ParameterSetDescription &desc);
  static const char *fillDescriptionsLabel() { return "ConsecutiveHits"; }

  // initialize the "event dependent state"
  void init(const TrackingRegion &region, const edm::EventSetup &es, const SeedComparitor *filter) final;

  // make job
  // fill seedCollection with the "TrajectorySeed"
  void makeSeed(TrajectorySeedCollection &seedCollection, const SeedingHitSet &hits) final;

private:
  virtual bool initialKinematic(GlobalTrajectoryParameters &kine, const SeedingHitSet &hits) const;

  bool checkHit(const TrajectoryStateOnSurface &tsos, SeedingHitSet::ConstRecHitPointer hit) const dso_hidden;

  CurvilinearTrajectoryError initialError(float sin2Theta) const dso_hidden;

  void buildSeed(TrajectorySeedCollection &seedCollection,
                 const SeedingHitSet &hits,
                 const FreeTrajectoryState &fts) const dso_hidden;

  SeedingHitSet::RecHitPointer refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                        const TrajectoryStateOnSurface &state) const dso_hidden;

protected:
  std::string thePropagatorLabel;
  float theBOFFMomentum;
  float theOriginTransverseErrorMultiplier;
  float theMinOneOverPtError;

  const TrackingRegion *region = nullptr;
  const SeedComparitor *filter = nullptr;
  TrackerGeometry const *trackerGeometry_;
  Propagator const *propagator_;
  MagneticField const *magneticField_;
  float nomField;
  bool isBOFF = false;
  std::string TTRHBuilder;
  std::string mfName_;
  bool forceKinematicWithRegionDirection_;

  TkClonerImpl cloner;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryESToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorESToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldESToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderESToken_;
};
#endif
