#ifndef MixedSeedLayerPairs_H
#define MixedSeedLayerPairs_H

/** \class MixedSeedLayerPairs
 * find all (resonable) pairs of pixel and strip layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"

class MixedSeedLayerPairs : public SeedLayerPairs{
public:
  MixedSeedLayerPairs():SeedLayerPairs(),isFirstCall(true),
    detLayersTIB(0),detLayersPosTID(0),detLayersNegTID(0),
    detLayersPosTEC(0),detLayersNegTEC(0),barrelLayers(0),
    fwdLayers(0),bkwLayers(0),allLayersWithHits(0){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);

  ~MixedSeedLayerPairs();

  //  virtual vector<LayerPair> operator()() const;
  vector<LayerPair> operator()() ;
  
  void init(const SiPixelRecHitCollection &collpxl,
	    const SiStripMatchedRecHit2DCollection &collmatch,
	    const SiStripRecHit2DCollection &collstereo, 
	    const SiStripRecHit2DCollection &collrphi,
	    const edm::EventSetup& iSetup);
  
private:
  TrackerLayerIdAccessor acc;
  
  bool isFirstCall;

  vector<BarrelDetLayer*>   detLayersPxlBarrel;
  vector<BarrelDetLayer*>   detLayersTIB;
  vector<ForwardDetLayer*>  detLayersPosPxl;
  vector<ForwardDetLayer*>  detLayersNegPxl;
  vector<ForwardDetLayer*>  detLayersPosTID;
  vector<ForwardDetLayer*>  detLayersNegTID;
  vector<ForwardDetLayer*>  detLayersPosTEC;
  vector<ForwardDetLayer*>  detLayersNegTEC;


 
  vector<LayerWithHits*> barrelLayers;
  vector<LayerWithHits*> fwdLayers;
  vector<LayerWithHits*> bkwLayers;
  vector<LayerWithHits*> allLayersWithHits;

 
  
 private:
  vector<const TrackingRecHit*> selectHitTIB(const SiStripMatchedRecHit2DCollection &collmatch,
					     const SiStripRecHit2DCollection &collstereo, 
					     const SiStripRecHit2DCollection &collrphi,
					     int tibNumber);
  
  vector<const TrackingRecHit*> selectHitTID(const SiStripMatchedRecHit2DCollection &collmatch,
					     const SiStripRecHit2DCollection &collstereo, 
					     const SiStripRecHit2DCollection &collrphi,
					     int side,
					     int disk,
					     int firstRing,
					     int lastRing);

  vector<const TrackingRecHit*> selectHitTEC(const SiStripMatchedRecHit2DCollection &collmatch,
					     const SiStripRecHit2DCollection &collstereo, 
					     const SiStripRecHit2DCollection &collrphi,
					     int side,
					     int disk,
					     int firstRing,
					     int lastRing);

  void addBarrelBarrelLayers( int mid, int outer, 
			      std::vector<LayerPair>& result) const;
  
  void addBarrelForwardLayers( int mid, int outer, 
			       std::vector<LayerPair>& result) const;

  void addForwardForwardLayers( int mid, int outer, 
				std::vector<LayerPair>& result) const;
  
};




#endif
