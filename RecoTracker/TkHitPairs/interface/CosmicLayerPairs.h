#ifndef CosmicLayerPairs_H
#define CosmicLayerPairs_H

/** \class CosmicLayerPairs
* find all (resonable) pairs of pixel layers
 */
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
//#include "RecoTracker/TkDetLayers/interface/PixelForwardLayer.h"
#include <vector>

class TrackerTopology;

class CosmicLayerPairs : public SeedLayerPairs{
public:
  CosmicLayerPairs(std::string geometry):_geometry(geometry){};//:isFirstCall(true){};
  ~CosmicLayerPairs();
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);



  //  virtual vector<LayerPair> operator()() const;
  std::vector<SeedLayerPairs::LayerPair> operator()() ;
  void init(const SiStripRecHit2DCollection &collstereo,
             const SiStripRecHit2DCollection &collrphi,
             const SiStripMatchedRecHit2DCollection &collmatched,
             //std::string geometry,
             const edm::EventSetup& iSetup);

private:

   //bool isFirstCall;
   std::string _geometry;

   std::vector<BarrelDetLayer const*> bl;
   std::vector<ForwardDetLayer const*> fpos;
   std::vector<ForwardDetLayer const*> fneg;
   edm::OwnVector<LayerWithHits> TECPlusLayerWithHits;
   edm::OwnVector<LayerWithHits> TECMinusLayerWithHits;
   edm::OwnVector<LayerWithHits> TIBLayerWithHits;
   edm::OwnVector<LayerWithHits> TOBLayerWithHits;
   edm::OwnVector<LayerWithHits> MTCCLayerWithHits;
   edm::OwnVector<LayerWithHits> CRACKLayerWithHits;

   std::vector<const TrackingRecHit*> selectTECHit(const SiStripRecHit2DCollection &collrphi,
                                                   const TrackerTopology& ttopo,
                                                                int side,
                                                                int disk);
   std::vector<const TrackingRecHit*> selectTIBHit(const SiStripRecHit2DCollection &collrphi,
                                                   const TrackerTopology& ttopo,
								int layer);
   std::vector<const TrackingRecHit*> selectTOBHit(const SiStripRecHit2DCollection &collrphi,
                                                   const TrackerTopology& ttopo,
                                                                int layer);
   std::vector<const TrackingRecHit*> selectTECHit(const SiStripMatchedRecHit2DCollection &collmatch,
                                                   const TrackerTopology& ttopo,
                                                                int side,
                                                                int disk);
   std::vector<const TrackingRecHit*> selectTIBHit(const SiStripMatchedRecHit2DCollection &collmatch,
                                                   const TrackerTopology& ttopo,
                                                                int layer);
   std::vector<const TrackingRecHit*> selectTOBHit(const SiStripMatchedRecHit2DCollection &collmatch,
                                                   const TrackerTopology& ttopo,
                                                                int layer);	
};




#endif
