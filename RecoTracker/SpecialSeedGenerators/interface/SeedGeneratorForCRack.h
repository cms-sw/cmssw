#ifndef SeedGeneratorForCRack_H
#define SeedGeneratorForCRack_H

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
class PixelSeedLayerPairs;

class SeedGeneratorForCRack{
 public:
  typedef TrajectoryStateOnSurface TSOS;
  SeedGeneratorForCRack(const edm::ParameterSet& conf);
  virtual ~SeedGeneratorForCRack(){};
  void init(const SiStripRecHit2DCollection &collstereo,
	    const SiStripRecHit2DCollection &collrphi,
	    const SiStripMatchedRecHit2DCollection &collmatched,
	    const edm::EventSetup& c);



  void  run(TrajectorySeedCollection &,const edm::EventSetup& c);
  void  seeds(TrajectorySeedCollection &output,
	      const edm::EventSetup& c,
	      const TrackingRegion& region);
 
 private:
  edm::ParameterSet conf_;
  GlobalTrackingRegion region;
  CosmicHitPairGenerator* thePairGenerator;
  edm::ESHandle<MagneticField> magfield;
  edm::ESHandle<TrackerGeometry> tracker;
  
  KFUpdator *theUpdator;
  PropagatorWithMaterial  *thePropagatorAl;
  PropagatorWithMaterial  *thePropagatorOp;
  const TransientTrackingRecHitBuilder *TTTRHBuilder;
  std::string builderName;
  std::string geometry;
  float seedpt;
  OrderedHitPairs HitPairs;
  float multipleScatteringFactor;
  double seedMomentum;
};
#endif


