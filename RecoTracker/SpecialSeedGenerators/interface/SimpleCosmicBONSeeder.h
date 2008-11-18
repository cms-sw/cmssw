#ifndef SimpleCosmicBONSeeder_h
#define SimpleCosmicBONSeeder_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

class SimpleCosmicBONSeeder : public edm::EDProducer
{
 public:

  explicit SimpleCosmicBONSeeder(const edm::ParameterSet& conf);

  virtual ~SimpleCosmicBONSeeder() {}

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  void init(const edm::EventSetup& c);
  void triplets(const edm::Event &e , const edm::EventSetup& c);
  void seedsInOut(TrajectorySeedCollection &output, const edm::EventSetup& iSetup);
  void seedsOutIn(TrajectorySeedCollection &output, const edm::EventSetup& iSetup);
  void seeds(TrajectorySeedCollection &output, const edm::EventSetup& iSetup);
  void done();

  bool goodTriplet(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer) const ;

  std::pair<GlobalVector,int>
  pqFromHelixFit(const GlobalPoint &inner, const GlobalPoint & middle, const GlobalPoint & outer, const edm::EventSetup& iSetup) const ;

 private:
  edm::ParameterSet conf_;
  std::string builderName;

  SeedingLayerSetsBuilder theLsb;
  GlobalTrackingRegion region;
  bool writeTriplets_;

  bool   seedOnMiddle_; 
  double rescaleError_;

  uint32_t tripletsVerbosity_,seedVerbosity_, helixVerbosity_;
 
  edm::ESHandle<MagneticField>                  magfield;
  edm::ESHandle<TrackerGeometry>                tracker;
  edm::ESHandle<TransientTrackingRecHitBuilder> TTTRHBuilder;
  KFUpdator               *theUpdator;
  PropagatorWithMaterial  *thePropagatorAl;
  PropagatorWithMaterial  *thePropagatorOp;
  TrajectoryStateTransform transformer;

  ClusterChecker check_;

  OrderedHitTriplets hitTriplets;

};

#endif
