#ifndef PixelSeedLayerPairs_H
#define PixelSeedLayerPairs_H

/** \class PixelSeedLayerPairs
 * find all (resonable) pairs of pixel layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"
class PixelSeedLayerPairs : public SeedLayerPairs{
public:
  PixelSeedLayerPairs():SeedLayerPairs(){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);



  //  virtual vector<LayerPair> operator()() const;
  vector<LayerPair> operator()() ;


private:

  //definition of the map 
 
  SiPixelRecHitCollection::range map_range1;
  SiPixelRecHitCollection::range map_range2;
  SiPixelRecHitCollection::range map_range3;

  SiPixelRecHitCollection::range map_diskneg1;
  SiPixelRecHitCollection::range map_diskneg2;

  SiPixelRecHitCollection::range map_diskpos1;
  SiPixelRecHitCollection::range map_diskpos2;

  TrackerLayerIdAccessor acc;
  
  LayerWithHits *lh1;
  LayerWithHits *lh2;
  LayerWithHits *lh3;

  LayerWithHits *pos1;
  LayerWithHits *pos2;

  LayerWithHits *neg1;
  LayerWithHits *neg2;


   vector<BarrelDetLayer*> bl;
   vector<ForwardDetLayer*> fpos;
   vector<ForwardDetLayer*> fneg;
 public:
 
   void init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup);
 private:
  void addBarrelBarrelLayers( int mid, int outer, 
       vector<LayerPair>& result) const;
  void addBarrelForwardLayers( int mid, int outer, 
       vector<LayerPair>& result) const ;
  void addForwardForwardLayers( int mid, int outer, 
       vector<LayerPair>& result) const;
};




#endif
