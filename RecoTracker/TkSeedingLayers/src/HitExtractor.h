#ifndef RecoTracker_TkSeedingLayers_HitExtractor_H
#define RecoTracker_TkSeedingLayers_HitExtractor_H

#include <vector>
#include <iterator>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"

namespace edm { class Event; class EventSetup; }
namespace ctfseeding { class SeedingLayer; }

namespace ctfseeding {

class HitExtractor {
public:
  virtual ~HitExtractor(){}
  virtual std::vector<SeedingHit> hits(const SeedingLayer & sl, const edm::Event& , const edm::EventSetup& ) const =0;
};

class Hit2SeedingHit {
    public:
        Hit2SeedingHit(const SeedingLayer &sl, const edm::EventSetup &es) : sl_(sl), es_(es) {}
        template<typename H> 
        SeedingHit operator()(const H &hit) { return SeedingHit(&hit, sl_, es_); }
    private:
        const SeedingLayer    &sl_;
        const edm::EventSetup &es_;
};

template <typename DSTV, typename A, typename B>
void range2SeedingHits(DSTV const & dstv,    
        std::vector<SeedingHit> & v, 
        std::pair<A,B> const & sel,
        const SeedingLayer &sl, const edm::EventSetup &es) {
    typename DSTV::Range range = dstv.equal_range(sel.first,sel.second);
    for(typename DSTV::const_iterator id=range.first; id!=range.second; id++){
        std::transform((*id).begin(), (*id).end(), std::back_inserter(v), Hit2SeedingHit(sl, es));
    } 
}

}




#endif
