#ifndef CosmicLayerPairs_H
#define CosmicLayerPairs_H

/** \class CosmicLayerPairs
 * find all (resonable) pairs of pixel layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
//#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"
class CosmicLayerPairs : public SeedLayerPairs{
public:
  CosmicLayerPairs():SeedLayerPairs(){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);



  //  virtual vector<LayerPair> operator()() const;
  vector<LayerPair> operator()() ;


private:

  //definition of the map 
 
  SiStripRecHit2DLocalPosCollection::range rphi_range1;
  SiStripRecHit2DLocalPosCollection::range rphi_range2;

  SiStripRecHit2DLocalPosCollection::range stereo_range1;
  SiStripRecHit2DLocalPosCollection::range stereo_range2;

  TrackerLayerIdAccessor acc;
  
  LayerWithHits *lh1;
  LayerWithHits *lh2;



   vector<BarrelDetLayer*> bl;
   vector<ForwardDetLayer*> fpos;
   vector<ForwardDetLayer*> fneg;
 public:
 
   void init(const SiStripRecHit2DLocalPosCollection &collstereo,
	     const SiStripRecHit2DLocalPosCollection &collrphi,
	     const edm::EventSetup& iSetup);

};




#endif
