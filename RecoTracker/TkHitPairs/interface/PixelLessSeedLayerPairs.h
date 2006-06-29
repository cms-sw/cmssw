#ifndef PixelLessSeedLayerPairs_H
#define PixelLessSeedLayerPairs_H

/** \class PixelLessSeedLayerPairs
 * find all (resonable) pairs of strip layers
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
#include "RecoTracker/TkDetLayers/interface/TIBLayer.h"

class PixelLessSeedLayerPairs : public SeedLayerPairs{
public:
  PixelLessSeedLayerPairs():SeedLayerPairs(),isFirstCall(true),
    detLayersTIB(0),detLayersPosTID(0),detLayersNegTID(0),
    detLayersPosTEC(0),detLayersNegTEC(0),barrelLayers(0),
    fwdLayers(0),bkwLayers(0),allLayersWithHits(0){};
  //  explicit PixelSeedLayerPairs(const edm::EventSetup& iSetup);

  ~PixelLessSeedLayerPairs();

  //  virtual vector<LayerPair> operator()() const;
  vector<LayerPair> operator()() ;

  void init(const SiStripRecHit2DMatchedLocalPosCollection &collmatch,
	    const SiStripRecHit2DLocalPosCollection &collstereo, 
	    const SiStripRecHit2DLocalPosCollection &collrphi,
	    const edm::EventSetup& iSetup);
  
private:
  TrackerLayerIdAccessor acc;
  
  bool isFirstCall;
  
  vector<BarrelDetLayer*>   detLayersTIB;
  vector<ForwardDetLayer*>  detLayersPosTID;
  vector<ForwardDetLayer*>  detLayersNegTID;
  vector<ForwardDetLayer*>  detLayersPosTEC;
  vector<ForwardDetLayer*>  detLayersNegTEC;


 
  vector<LayerWithHits*> barrelLayers;
  vector<LayerWithHits*> fwdLayers;
  vector<LayerWithHits*> bkwLayers;
  vector<LayerWithHits*> allLayersWithHits;

 
  
 private:
  vector<const TrackingRecHit*> selectHitTIB(const SiStripRecHit2DMatchedLocalPosCollection &collmatch,
					     const SiStripRecHit2DLocalPosCollection &collstereo, 
					     const SiStripRecHit2DLocalPosCollection &collrphi,
					     int tibNumber);
  
  vector<const TrackingRecHit*> selectHitTID(const SiStripRecHit2DMatchedLocalPosCollection &collmatch,
					     const SiStripRecHit2DLocalPosCollection &collstereo, 
					     const SiStripRecHit2DLocalPosCollection &collrphi,
					     int side,
					     int disk,
					     int firstRing,
					     int lastRing);

  vector<const TrackingRecHit*> selectHitTEC(const SiStripRecHit2DMatchedLocalPosCollection &collmatch,
					     const SiStripRecHit2DLocalPosCollection &collstereo, 
					     const SiStripRecHit2DLocalPosCollection &collrphi,
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
