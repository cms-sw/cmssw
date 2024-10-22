#ifndef SeedGeneratorForCosmics_H
#define SeedGeneratorForCosmics_H

/** \class CombinatorialSeedGeneratorFromPixel
 *  A concrete regional seed generator providing seeds constructed 
 *  from combinations of hits in pairs of pixel layers 
 */
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
//#include "RecoTracker/SpecialSeedGenerators/interface/SeedGeneratorFromLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGenerator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/PixelSeeding/interface/CosmicHitTripletGenerator.h"
class PixelSeedLayerPairs;
class GeometricSearchTracker;
class TrackerRecoGeometryRecord;

class SeedGeneratorForCosmics {
public:
  typedef TrajectoryStateOnSurface TSOS;
  SeedGeneratorForCosmics(const edm::ParameterSet &conf, edm::ConsumesCollector);

  void run(const SiStripRecHit2DCollection &collstereo,
           const SiStripRecHit2DCollection &collrphi,
           const SiStripMatchedRecHit2DCollection &collmatched,
           const edm::EventSetup &c,
           TrajectorySeedCollection &);

private:
  void init(const SiStripRecHit2DCollection &collstereo,
            const SiStripRecHit2DCollection &collrphi,
            const SiStripMatchedRecHit2DCollection &collmatched,
            const edm::EventSetup &c);

  bool seeds(TrajectorySeedCollection &output, const TrackingRegion &region);

  int32_t maxSeeds_;
  GlobalTrackingRegion region;
  CosmicHitPairGenerator *thePairGenerator;
  CosmicHitTripletGenerator *theTripletGenerator;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMagfieldToken;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> theTrackerToken;
  const edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> theSearchTrackerToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> theTTopoToken;

  KFUpdator *theUpdator;
  PropagatorWithMaterial *thePropagatorAl;
  PropagatorWithMaterial *thePropagatorOp;
  std::string geometry;
  std::string hitsforseeds;
  float seedpt;
  OrderedHitPairs HitPairs;
  OrderedHitTriplets HitTriplets;

  //***top-bottom
  bool positiveYOnly;
  bool negativeYOnly;
  //***
};
#endif
