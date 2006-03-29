#ifndef PixelLessSeedLayerPairs_H
#define PixelLessSeedLayerPairs_H

/** \class PixelLessSeedLayerPairs
 * find all (resonable) pairs of pixel layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
//#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"
class PixelLessSeedLayerPairs : public SeedLayerPairs{
public:
  PixelLessSeedLayerPairs():SeedLayerPairs(){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);



  //  virtual vector<LayerPair> operator()() const;
  vector<LayerPair> operator()() ;


private:

  //definition of the map 
 
  SiStripRecHit2DMatchedLocalPosCollection::range map_range1;
  SiStripRecHit2DMatchedLocalPosCollection::range map_range2;


  TrackerLayerIdAccessor acc;
  
  LayerWithHits *lh1;
  LayerWithHits *lh2;
/*   LayerWithHits *lh3; */

/*   LayerWithHits *pos1; */
/*   LayerWithHits *pos2; */

/*   LayerWithHits *neg1; */
/*   LayerWithHits *neg2; */


   vector<BarrelDetLayer*> bl;
   vector<ForwardDetLayer*> fpos;
   vector<ForwardDetLayer*> fneg;
 public:
 
   void init(const SiStripRecHit2DMatchedLocalPosCollection &collmatch,
	     const SiStripRecHit2DLocalPosCollection &collrphi,
	     const edm::EventSetup& iSetup);
 private:
  void addBarrelBarrelLayers( int mid, int outer, 
       vector<LayerPair>& result) const;
  void addBarrelForwardLayers( int mid, int outer, 
       vector<LayerPair>& result) const ;
  void addForwardForwardLayers( int mid, int outer, 
       vector<LayerPair>& result) const;
};




#endif
