#ifndef LayerHitMapCache_H
#define LayerHitMapCache_H

/** A cache adressable by DetLayer* and TrackingRegion* .
 *  Used to cache all the hits of a DetLayer.
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"


class LayerHitMapCache {

private:

  template <class KeyType, class ValueType> class SimpleCache {
  public:
    SimpleCache(int initSize) { reserve(initSize); }
    virtual ~SimpleCache() { clear(); }
    void reserve(int size) { theContainer.reserve(size); }
    const ValueType*  get(const KeyType & key) {
      for (ConstItr it = theContainer.begin(); it != theContainer.end(); it++) {
        if ( it->first == key) return it->second;
      }
      return 0;
    }
    /// add object to cache. It is caller responsibility to check that object is not yet there.
    void add(const KeyType & key, const ValueType * value) {
      theContainer.push_back( std::make_pair(key,value));
    }
    /// emptify cache, delete values associated to Key
    virtual void clear() {
      for (ConstItr i=theContainer.begin(); i!= theContainer.end(); i++) { delete i->second; }
      theContainer.clear();
    }
  protected:
    typedef std::pair< KeyType, const ValueType * > KeyValuePair;
    std::vector< KeyValuePair > theContainer;
    typedef typename std::vector< KeyValuePair >::const_iterator ConstItr;
  private:
    SimpleCache(const SimpleCache &) { }
  };

private:
  typedef const DetLayer * LayerRegionKey;
  typedef SimpleCache<LayerRegionKey, RecHitsSortedInPhi> Cache;
 public:
  LayerHitMapCache(int initSize=50) { theCache = new Cache(initSize); }

  ~LayerHitMapCache() { delete theCache; }

  void clear() { theCache->clear(); }

  const RecHitsSortedInPhi & operator()(
      const ctfseeding::SeedingLayer * layer, const TrackingRegion & region, 
      const edm::Event & iEvent, const edm::EventSetup & iSetup) {
    LayerRegionKey key(layer->detLayer());
    const RecHitsSortedInPhi * lhm = theCache->get(key);
    if (lhm==0) {
      lhm=new RecHitsSortedInPhi (region.hits(iEvent,iSetup,layer));
      LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits in the cache for: "<<layer->detLayer();
      theCache->add( key, lhm); 
    }
    else{
      LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits FROM THE cache for: "<<layer->detLayer();
    }
    return *lhm;
  }

public:
  LayerHitMapCache(const LayerHitMapCache &) { }

private:
  Cache * theCache; 
};

#endif

