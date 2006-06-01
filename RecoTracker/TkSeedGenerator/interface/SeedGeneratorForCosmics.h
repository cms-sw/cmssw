#ifndef SeedGeneratorForCosmics_H
#define SeedGeneratorForCosmics_H

/** \class CombinatorialSeedGeneratorFromPixel
 *  A concrete regional seed generator providing seeds constructed 
 *  from combinations of hits in pairs of pixel layers 
 */
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"    
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromLayerPairs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGenerator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
class PixelSeedLayerPairs;

class SeedGeneratorForCosmics : public SeedGeneratorFromTrackingRegion {
 public:
  typedef TrajectoryStateOnSurface TSOS;
  SeedGeneratorForCosmics(const edm::ParameterSet& conf);
  virtual ~SeedGeneratorForCosmics(){};
  void init(const SiStripRecHit2DLocalPosCollection &collstereo,
	    const SiStripRecHit2DLocalPosCollection &collrphi,
	    const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
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
  TrajectoryStateTransform transformer;
  KFUpdator *theUpdator;
  PropagatorWithMaterial  *thePropagatorAl;
  PropagatorWithMaterial  *thePropagatorOp;
  TkTransientTrackingRecHitBuilder *TTTRHBuilder;
};
#endif


