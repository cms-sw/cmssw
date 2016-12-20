#ifndef LayerHitMapCache_H
#define LayerHitMapCache_H

/** A cache adressable by DetLayer* and TrackingRegion* .
 *  Used to cache all the hits of a DetLayer.
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Framework/interface/EventSetup.h"


class LayerHitMapCache {

private:

  class SimpleCache {
  public:
    using ValueType = RecHitsSortedInPhi;
    using KeyType = int;
    SimpleCache(unsigned int initSize) : theContainer(initSize, nullptr){}
    ~SimpleCache() { clear(); }
    void resize(int size) { theContainer.resize(size,nullptr); }
    const ValueType*  get(KeyType key) { return theContainer[key];}
    /// add object to cache. It is caller responsibility to check that object is not yet there.
    void add(KeyType key, const ValueType * value) {
      if (key>=int(theContainer.size())) resize(key+1);
      theContainer[key]=value;
    }
    /// emptify cache, delete values associated to Key
    void clear() {      
      for ( auto & v : theContainer)  { delete v; v=nullptr;}
    }
  private:
    std::vector< const ValueType *> theContainer;
  private:
    SimpleCache(const SimpleCache &) { }
  };

private:
  typedef SimpleCache Cache;
public:
  LayerHitMapCache(unsigned int initSize=50) : theCache(initSize) { }

  void clear() { theCache.clear(); }
  
  const RecHitsSortedInPhi &
  operator()(const SeedingLayerSetsHits::SeedingLayer& layer, const TrackingRegion & region,
	     const edm::Event & iEvent, const edm::EventSetup & iSetup) {
    int key = layer.index();
    assert (key>=0);
    const RecHitsSortedInPhi * lhm = theCache.get(key);
    if (lhm==nullptr) {
      lhm=new RecHitsSortedInPhi (region.hits(iEvent,iSetup,layer), region.origin(), layer.detLayer());
      lhm->theOrigin = region.origin();
      LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits in the cache for: "<<layer.detLayer();
      theCache.add( key, lhm);
    }
    else{
      // std::cout << region.origin() << " " <<  lhm->theOrigin << std::endl;
      LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits FROM THE cache for: "<<layer.detLayer();
    }
    return *lhm;
  }

private:
  Cache theCache; 
};

#endif

