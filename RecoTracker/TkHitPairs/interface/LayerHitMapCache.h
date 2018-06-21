#ifndef LayerHitMapCache_H
#define LayerHitMapCache_H

/** A cache adressable by DetLayer* and TrackingRegion* .
 *  Used to cache all the hits of a DetLayer.
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"

class LayerHitMapCache {

private:

  class SimpleCache {
  public:
    using ValueType = RecHitsSortedInPhi;
    using KeyType = int;
    SimpleCache(unsigned int initSize) : theContainer(initSize){}
    SimpleCache(const SimpleCache&) = delete;
    SimpleCache& operator=(const SimpleCache&) = delete;
    SimpleCache(SimpleCache&&) = default;
    SimpleCache& operator=(SimpleCache&&) = default;
    ~SimpleCache() { clear(); }
    void resize(int size) { theContainer.resize(size); }
    const ValueType*  get(KeyType key) const { return theContainer[key].get();}
    /// add object to cache. It is caller responsibility to check that object is not yet there.
    void add(KeyType key, ValueType * value) {
      if (key>=int(theContainer.size())) resize(key+1);
      theContainer[key].reset(value);
    }
    void extend(const SimpleCache& other) {
      // N.B. Here we assume that the lifetime of 'other' is longer than of 'this'.
      if(other.theContainer.size() > theContainer.size())
        resize(other.theContainer.size());

      for(size_t i=0, size=other.theContainer.size(); i != size; ++i) {
        assert(get(i) == nullptr); // We don't want to override any existing value
        theContainer[i].reset(*(other.get(i))); // pass by reference to denote that we don't own it
      }
    }
    /// emptify cache, delete values associated to Key
    void clear() {      
      for ( auto & v : theContainer)  { v.reset(); }
    }
  private:
    std::vector<mayown_ptr<ValueType> > theContainer;
  };

private:
  typedef SimpleCache Cache;
public:
  LayerHitMapCache(unsigned int initSize=50) : theCache(initSize) { }
  LayerHitMapCache(LayerHitMapCache&&) = default;
  LayerHitMapCache& operator=(LayerHitMapCache&&) = default;
  

  void clear() { theCache.clear(); }

  void extend(const LayerHitMapCache& other) {
    theCache.extend(other.theCache);
  }

  // Mainly for FastSim, overrides old hits if exists
  RecHitsSortedInPhi *add(const SeedingLayerSetsHits::SeedingLayer& layer, std::unique_ptr<RecHitsSortedInPhi> hits) {
    RecHitsSortedInPhi *ptr = hits.get();
    theCache.add(layer.index(), hits.release());
    return ptr;
  }
  
  const RecHitsSortedInPhi &
  operator()(const SeedingLayerSetsHits::SeedingLayer& layer, const TrackingRegion & region,
	     const edm::EventSetup & iSetup) {
    int key = layer.index();
    assert (key>=0);
    const RecHitsSortedInPhi * lhm = theCache.get(key);
    if (lhm==nullptr) {
      auto tmp = add(layer, std::make_unique<RecHitsSortedInPhi>(region.hits(iSetup,layer), region.origin(), layer.detLayer()));
      tmp->theOrigin = region.origin();
      lhm = tmp;
      LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits in the cache for: "<<layer.detLayer();
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

