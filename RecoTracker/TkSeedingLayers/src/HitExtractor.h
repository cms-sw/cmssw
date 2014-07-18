#ifndef RecoTracker_TkSeedingLayers_HitExtractor_H
#define RecoTracker_TkSeedingLayers_HitExtractor_H

#include <vector>
#include <iterator>
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ContainerMask.h"

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"


namespace edm { class Event; class EventSetup; class ConsumesCollector;}
namespace ctfseeding { class SeedingLayer; }

namespace ctfseeding {

  class HitExtractor {
  public:
    using TkHit = BaseTrackerRecHit;
    using TkHitRef = BaseTrackerRecHit const &;
    using HitPointer = mayown_ptr<BaseTrackerRecHit>;
    using Hits=std::vector<HitPointer>;
    
    virtual ~HitExtractor(){}
    HitExtractor() : skipClusters(false){}
    
    virtual Hits hits(const TkTransientTrackingRecHitBuilder& ttrhBuilder, const edm::Event& , const edm::EventSetup& ) const =0;
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
  
  
  template <typename DSTV, typename A, typename B>
  inline void range2SeedingHits(DSTV const & dstv,
				HitExtractor::Hits & v,
				std::pair<A,B> const & sel) {
    typename DSTV::Range range = dstv.equal_range(sel.first,sel.second);
    size_t ts = v.size();
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++)
      ts += std::distance((*id).begin(), (*id).end());
    v.reserve(ts);
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++){
      for ( auto const & h : (*id) ) v.emplace_back(h);
    }
  
  }
}
#endif
