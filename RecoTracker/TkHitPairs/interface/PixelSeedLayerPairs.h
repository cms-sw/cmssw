#ifndef PixelSeedLayerPairs_H
#define PixelSeedLayerPairs_H

/** \class PixelSeedLayerPairs
 * find all (resonable) pairs of pixel layers
 */
#include "Geometry/TrackerSimAlgo/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"

class PixelSeedLayerPairs : public SeedLayerPairs{
public:
  PixelSeedLayerPairs():SeedLayerPairs(){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);



  //  virtual vector<LayerPair> operator()() const;
  vector<LayerPair> operator()() ;
/*   const LayerAccessor::BarrelLayerContainer& barrelPixel() const {return theBarrelPixel;} */
/*   const LayerAccessor::ForwardLayerContainer& negativePixel() const {return theNegPixel;} */
/*   const LayerAccessor::ForwardLayerContainer& positivePixel() const {return thePosPixel;} */

private:

  //definition of the map 
   edm::RangeMap<DetId, edm::OwnVector<SiPixelRecHitCollection,edm::ClonePolicy<SiPixelRecHitCollection> >, edm::ClonePolicy<SiPixelRecHitCollection> > map;
   //definition of the range
   edm::RangeMap<DetId, edm::OwnVector<SiPixelRecHitCollection,edm::ClonePolicy<SiPixelRecHitCollection> >, edm::ClonePolicy<SiPixelRecHitCollection> >::range  map_range1; 
  edm::RangeMap<DetId, edm::OwnVector<SiPixelRecHitCollection,edm::ClonePolicy<SiPixelRecHitCollection> >, edm::ClonePolicy<SiPixelRecHitCollection> >::range  map_range2; 
  
   TrackerLayerIdAccessor acc;
   TrackerLayerIdAccessor::returnType lay1;
   TrackerLayerIdAccessor::returnType lay2;
   TrackerLayerIdAccessor::returnType lay3;
   TrackerLayerIdAccessor::returnType dr1;
   TrackerLayerIdAccessor::returnType dr2;
   TrackerLayerIdAccessor::returnType dr3;
    //  const SiPixelRecHitCollection::Range lay1range;

  /*   LayerAccessor::BarrelLayerContainer  theBarrelPixel; */
/*   LayerAccessor::ForwardLayerContainer theNegPixel; */
/*   LayerAccessor::ForwardLayerContainer thePosPixel; */
   vector<BarrelDetLayer*> bl;

 public:
 
   void init(const edm::EventSetup& iSetup);
 private:
  void addBarrelBarrelLayers( int mid, int outer, 
       vector<LayerPair>& result) const;
  void addBarrelForwardLayers( int mid, int outer, 
       vector<LayerPair>& result) const ;
  void addForwardForwardLayers( int mid, int outer, 
       vector<LayerPair>& result) const;
};




#endif
