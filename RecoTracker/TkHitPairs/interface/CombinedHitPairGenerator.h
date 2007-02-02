#ifndef CombinedHitPairGenerator_H
#define CombinedHitPairGenerator_H

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
class HitPairGeneratorFromLayerPair;


/** \class CombinedHitPairGenerator
 * Hides set of HitPairGeneratorFromLayerPair generators.
 */

class CombinedHitPairGenerator : public HitPairGenerator{

  typedef std::vector<HitPairGeneratorFromLayerPair *>   Container;

public:
  CombinedHitPairGenerator(SeedLayerPairs& layers, const edm::EventSetup& iSetup);
  CombinedHitPairGenerator(SeedLayerPairs& layers);
  typedef LayerHitMapCache LayerCacheType;

  
  //  CombinedHitPairGenerator(const SeedLayerPairs & layers);
  

  // copy configuration but empty cache
  // CombinedHitPairGenerator(const CombinedHitPairGenerator&);

  ~CombinedHitPairGenerator();

  /// add generators based on layers
    //  void  add(const DetLayer* inner, const DetLayer* outer);
    void  add(const LayerWithHits* inner, 
	      const LayerWithHits* outer,
	      const edm::EventSetup& iSetup);
  /// form base class
  virtual void hitPairs( const TrackingRegion& reg, 
			 OrderedHitPairs & prs, 
			 const edm::EventSetup& iSetup);

  /// from base class
  virtual CombinedHitPairGenerator * clone() const 
    { return new CombinedHitPairGenerator(*this); }

private:

  LayerCacheType   theLayerCache;
  Container        theGenerators;

};
#endif
