#ifndef RecoTracker_TkSeedingLayers_HitExtractor_H
#define RecoTracker_TkSeedingLayers_HitExtractor_H

#include <vector>
#include <iterator>
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ContainerMask.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

namespace edm { class Event; class EventSetup; class ConsumesCollector;}
namespace ctfseeding { class SeedingLayer; }

namespace ctfseeding {

class HitExtractor {
public:
  typedef std::vector<TransientTrackingRecHit::ConstRecHitPointer> Hits;
  virtual ~HitExtractor(){}
  HitExtractor(){
    skipClusters=false;}
  virtual Hits hits(const TransientTrackingRecHitBuilder& ttrhBuilder, const edm::Event& , const edm::EventSetup& ) const =0;
  virtual HitExtractor * clone() const = 0;

  //skip clusters
  void useSkipClusters(const edm::InputTag & m, edm::ConsumesCollector& iC) {
    skipClusters=true;
    useSkipClusters_(m, iC);
  }
  bool skipClusters;
protected:
  virtual void useSkipClusters_(const edm::InputTag & m, edm::ConsumesCollector& iC) = 0;
};

class HitConv {
public:
  HitConv(const TransientTrackingRecHitBuilder &ttrhBuilder, const edm::EventSetup &es) : ttrhBuilder_(ttrhBuilder), es_(es) {}
  template<typename H> 
  TransientTrackingRecHit::ConstRecHitPointer operator()(const H &hit) {
    const TrackingRecHit* trh = &hit;
    return ttrhBuilder_.build(trh); }
private:
  const TransientTrackingRecHitBuilder    &ttrhBuilder_;
  const edm::EventSetup &es_;

};
  
  template <typename DSTV, typename A, typename B>
  inline void range2SeedingHits(DSTV const & dstv,
				HitExtractor::Hits & v,
				std::pair<A,B> const & sel,
				const TransientTrackingRecHitBuilder &ttrhBuilder, const edm::EventSetup &es) {
    typename DSTV::Range range = dstv.equal_range(sel.first,sel.second);
    size_t ts = v.size();
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++)
      ts += std::distance((*id).begin(), (*id).end());
    v.reserve(ts);
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++){
      std::transform((*id).begin(), (*id).end(), std::back_inserter(v), HitConv(ttrhBuilder, es));
    }
  }
  
}

#endif
