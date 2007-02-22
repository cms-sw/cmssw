#ifndef PixelLessSeedLayerPairs_H
#define PixelLessSeedLayerPairs_H

/** \class PixelLessSeedLayerPairs
 * find all (resonable) pairs of strip layers
 */
#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"

#include <vector>

class PixelLessSeedLayerPairs : public SeedLayerPairs{
public:
  PixelLessSeedLayerPairs():SeedLayerPairs(),isFirstCall(true),
    detLayersTIB(0),detLayersPosTID(0),detLayersNegTID(0),
    detLayersPosTEC(0),detLayersNegTEC(0),barrelLayers(0),
    fwdLayers(0),bkwLayers(0),allLayersWithHits(0){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);

  ~PixelLessSeedLayerPairs();

  //  virtual std::vector<LayerPair> operator()() const;
  std::vector<LayerPair> operator()() ;

  void init(const SiStripMatchedRecHit2DCollection &collmatch,
	    const SiStripRecHit2DCollection &collstereo, 
	    const SiStripRecHit2DCollection &collrphi,
	    const edm::EventSetup& iSetup);
  
private:
  TrackerLayerIdAccessor acc;
  
  bool isFirstCall;
  
  std::vector<BarrelDetLayer*>   detLayersTIB;
  std::vector<ForwardDetLayer*>  detLayersPosTID;
  std::vector<ForwardDetLayer*>  detLayersNegTID;
  std::vector<ForwardDetLayer*>  detLayersPosTEC;
  std::vector<ForwardDetLayer*>  detLayersNegTEC;


 
  std::vector<LayerWithHits*> barrelLayers;
  std::vector<LayerWithHits*> fwdLayers;
  std::vector<LayerWithHits*> bkwLayers;
  std::vector<LayerWithHits*> allLayersWithHits;

 
  
 private:
  std::vector<const TrackingRecHit*> selectHitTIB(const SiStripMatchedRecHit2DCollection &collmatch,
					     const SiStripRecHit2DCollection &collstereo, 
					     const SiStripRecHit2DCollection &collrphi,
					     int tibNumber);
  
  std::vector<const TrackingRecHit*> selectHitTID(const SiStripMatchedRecHit2DCollection &collmatch,
					     const SiStripRecHit2DCollection &collstereo, 
					     const SiStripRecHit2DCollection &collrphi,
					     int side,
					     int disk,
					     int firstRing,
					     int lastRing);

  std::vector<const TrackingRecHit*> selectHitTEC(const SiStripMatchedRecHit2DCollection &collmatch,
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
