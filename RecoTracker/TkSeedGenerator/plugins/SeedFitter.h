#ifndef RecoTracker_TkSeedGenerator_SeedFitter_H
#define RecoTracker_TkSeedGenerator_SeedFitter_H

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

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

class dso_hidden SeedFitter : public edm::stream::EDProducer<> {
public:
  SeedFitter(const edm::ParameterSet &);

  ~SeedFitter() override;

  static void fillDescription(edm::ConfigurationDescriptions &description);

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override;

  // initialize the "event dependent state"
  void init(const edm::EventSetup &es);

  // make job
  // fill seedCollection with the "ElectronSeedCollection"
  void makeSeed(reco::ElectronSeedCollection &seedCollection, const reco::ElectronSeed &seed);

private:
  SeedFitter(const edm::ParameterSet &, edm::ConsumesCollector &&);

  void initialKinematic(GlobalTrajectoryParameters &kine, const reco::ElectronSeed &seed) const;

  CurvilinearTrajectoryError initialError(float sin2Theta) const dso_hidden;

  void buildSeed(reco::ElectronSeedCollection &seedCollection,
                 const reco::ElectronSeed &seed,
                 const FreeTrajectoryState &fts) const dso_hidden;

  SeedingHitSet::RecHitPointer refitHit(SeedingHitSet::ConstRecHitPointer hit,
                                        const TrajectoryStateOnSurface &state) const dso_hidden;

protected:
  edm::EDGetTokenT<reco::ElectronSeedCollection> eleSeedCollectionToken_;

  std::string thePropagatorLabel_;
  float theBOFFMomentum_;
  float theOriginTransverseErrorMultiplier_;
  float theMinOneOverPtError_;

  TrackerGeometry const *trackerGeometry_;
  Propagator const *propagator_;
  MagneticField const *magneticField_;
  float nomField_ = 0.;
  bool isBOFF_ = false;
  std::string TTRHBuilder_;
  std::string mfName_;
  bool forceKinematicWithRegionDirection_;

  TkClonerImpl cloner_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometryESToken_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorESToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldESToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> transientTrackingRecHitBuilderESToken_;
};
#endif
