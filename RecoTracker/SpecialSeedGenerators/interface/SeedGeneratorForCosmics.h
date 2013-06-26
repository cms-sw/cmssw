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
#include "RecoPixelVertexing/PixelTriplets/interface/CosmicHitTripletGenerator.h"
class PixelSeedLayerPairs;

class SeedGeneratorForCosmics{
 public:
  typedef TrajectoryStateOnSurface TSOS;
  SeedGeneratorForCosmics(const edm::ParameterSet& conf);
  virtual ~SeedGeneratorForCosmics(){};
  void init(const SiStripRecHit2DCollection &collstereo,
	    const SiStripRecHit2DCollection &collrphi,
	    const SiStripMatchedRecHit2DCollection &collmatched,
	    const edm::EventSetup& c);



  void  run(TrajectorySeedCollection &,const edm::EventSetup& c);
  bool  seeds(TrajectorySeedCollection &output,
	      const edm::EventSetup& c,
	      const TrackingRegion& region);
 
 private:
  edm::ParameterSet conf_;
  int32_t           maxSeeds_;
  GlobalTrackingRegion region;
  CosmicHitPairGenerator* thePairGenerator;
  CosmicHitTripletGenerator* theTripletGenerator; 
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;
  
  KFUpdator *theUpdator;
  PropagatorWithMaterial  *thePropagatorAl;
  PropagatorWithMaterial  *thePropagatorOp;
  const TransientTrackingRecHitBuilder *TTTRHBuilder;
  std::string builderName;
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


