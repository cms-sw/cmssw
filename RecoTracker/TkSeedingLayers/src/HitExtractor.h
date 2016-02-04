#ifndef RecoTracker_TkSeedingLayers_HitExtractor_H
#define RecoTracker_TkSeedingLayers_HitExtractor_H

#include <vector>
#include <iterator>
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

namespace edm { class Event; class EventSetup; }
namespace ctfseeding { class SeedingLayer; }

namespace ctfseeding {

class HitExtractor {
public:
  typedef std::vector<TransientTrackingRecHit::ConstRecHitPointer> Hits;
  virtual ~HitExtractor(){}
  HitExtractor(){
    skipClusters=false;}
  virtual Hits hits(const SeedingLayer & sl, const edm::Event& , const edm::EventSetup& ) const =0;

  //skip clusters
  void useSkipClusters( const edm::InputTag & m) {
    skipClusters=true;
    theSkipClusters=m;
  }
  bool skipClusters;
  edm::InputTag theSkipClusters;
};

class HitConv {
public:
  HitConv(const SeedingLayer &sl, const edm::EventSetup &es) : sl_(sl), es_(es) {}
  template<typename H> 
  TransientTrackingRecHit::ConstRecHitPointer operator()(const H &hit) {
    const TrackingRecHit* trh = &hit;
    return sl_.hitBuilder()->build(trh); }
private:
  const SeedingLayer    &sl_;
  const edm::EventSetup &es_;

};
  
  template <typename DSTV, typename A, typename B>
  inline void range2SeedingHits(DSTV const & dstv,
				HitExtractor::Hits & v,
				std::pair<A,B> const & sel,
				const SeedingLayer &sl, const edm::EventSetup &es) {
    typename DSTV::Range range = dstv.equal_range(sel.first,sel.second);
    size_t ts = v.size();
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++)
      ts += std::distance((*id).begin(), (*id).end());
    v.reserve(ts);
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++){
      std::transform((*id).begin(), (*id).end(), std::back_inserter(v), HitConv(sl,es));
    }
  }
  
}

#endif
