#ifndef CombinedHitPairGeneratorBC_H
#define CombinedHitPairGeneratorBC_H

#include <vector>
#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"
class SeedLayerPairs;
class LayerWithHits;
class DetLayer;
class TrackingRegion;
class HitPairGeneratorFromLayerPairBC;


/** \class CombinedHitPairGeneratorBC
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CombinedHitPairGeneratorBC : public HitPairGenerator{

  typedef std::vector<HitPairGeneratorFromLayerPairBC *>   Container;

public:
  CombinedHitPairGeneratorBC(SeedLayerPairs& layers, const edm::EventSetup& iSetup);
  CombinedHitPairGeneratorBC(SeedLayerPairs& layers);
  typedef LayerHitMapCacheBC LayerCacheType;

  
  //  CombinedHitPairGeneratorBC(const SeedLayerPairs & layers);
  

  // copy configuration but empty cache
  // CombinedHitPairGeneratorBC(const CombinedHitPairGeneratorBC&);

  ~CombinedHitPairGeneratorBC();

  /// add generators based on layers
    void  add(const LayerWithHits* inner, 
	      const LayerWithHits* outer,
	      const edm::EventSetup& iSetup);
  /// form base class
  virtual void hitPairs( const TrackingRegion& reg, 
			 OrderedHitPairs & prs, 
			 const edm::EventSetup& iSetup);
  virtual void hitPairs( const TrackingRegion& reg, 
			 OrderedHitPairs & prs, 
                   const edm::Event& ev,
			 const edm::EventSetup& iSetup) {}

  /// from base class
  virtual CombinedHitPairGeneratorBC * clone() const 
    { return new CombinedHitPairGeneratorBC(*this); }

private:

  LayerCacheType   theLayerCache;
  Container        theGenerators;

};
#endif
