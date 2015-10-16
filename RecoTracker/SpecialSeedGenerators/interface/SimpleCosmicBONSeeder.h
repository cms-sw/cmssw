#ifndef SimpleCosmicBONSeeder_h
#define SimpleCosmicBONSeeder_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

class SeedingLayerSetsHits;

class SimpleCosmicBONSeeder : public edm::stream::EDProducer<>
{
 public:

  explicit SimpleCosmicBONSeeder(const edm::ParameterSet& conf);

  virtual ~SimpleCosmicBONSeeder() {}

  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;

  void init(const edm::EventSetup& c);
  bool triplets(const edm::Event &e , const edm::EventSetup& c);
  bool seeds(TrajectorySeedCollection &output, const edm::EventSetup& iSetup);
  void done();

  bool goodTriplet(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer, const double & minRho) const ;

  std::pair<GlobalVector,int>
  pqFromHelixFit(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer, const edm::EventSetup& iSetup) const ;

 private:
  edm::ParameterSet conf_;
  std::string builderName;

  edm::EDGetTokenT<SeedingLayerSetsHits> seedingLayerToken_;
  GlobalTrackingRegion region_;
  double pMin_;
  bool writeTriplets_;

  bool   seedOnMiddle_; 
  double rescaleError_;

  uint32_t tripletsVerbosity_,seedVerbosity_, helixVerbosity_;
 
  edm::ESHandle<MagneticField>                  magfield;
  edm::ESHandle<TrackerGeometry>                tracker;
  edm::ESHandle<TransientTrackingRecHitBuilder> TTTRHBuilder;
  TkClonerImpl cloner; // FIXME
  KFUpdator               *theUpdator;
  PropagatorWithMaterial  *thePropagatorAl;
  PropagatorWithMaterial  *thePropagatorOp;
  

  ClusterChecker check_;
  int32_t        maxTriplets_, maxSeeds_;

  OrderedHitTriplets hitTriplets;

  int  goodHitsPerSeed_; // number of hits that must be good
  bool checkCharge_;     // check cluster charge
  bool matchedRecHitUsesAnd_;
  std::vector<int32_t> chargeThresholds_;
  bool checkMaxHitsPerModule_;
  std::vector<int32_t> maxHitsPerModule_;
  bool checkCharge(const TrackingRecHit *hit) const ;
  bool checkCharge(const SiStripRecHit2D &hit, int subdetid) const ;
  void checkNoisyModules(const std::vector<SeedingHitSet::ConstRecHitPointer> &hits, std::vector<bool> &oks) const ;

  //***top-bottom
  bool positiveYOnly;
  bool negativeYOnly;
  //***
};

#endif
