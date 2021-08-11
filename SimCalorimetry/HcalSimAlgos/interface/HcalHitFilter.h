#ifndef HcalSimAlgos_HcalHitFilter_h
#define HcalSimAlgos_HcalHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <algorithm>
#include <vector>

template <HcalSubdetector... subdets>
class HcalHitFilter : public CaloVHitFilter {
public:
  HcalHitFilter() : theSubdets({subdets...}) { std::sort(theSubdets.begin(), theSubdets.end()); }
  ~HcalHitFilter() override {}

  void setDetIds(const std::vector<DetId>& detIds) {
    theDetIds = detIds;
    std::sort(theDetIds.begin(), theDetIds.end());
  }

  bool accepts(const PCaloHit& hit) const override {
    HcalDetId hcalDetId(hit.id());
    return ((theSubdets.empty() || std::binary_search(theSubdets.begin(), theSubdets.end(), hcalDetId.subdet())) &&
            (theDetIds.empty() || std::binary_search(theDetIds.begin(), theDetIds.end(), DetId(hit.id()))));
  }

protected:
  std::vector<HcalSubdetector> theSubdets;
  // empty DetIds will always be accepted
  std::vector<DetId> theDetIds;
};

typedef HcalHitFilter<HcalBarrel, HcalEndcap> HBHEHitFilter;
typedef HcalHitFilter<HcalForward> HFHitFilter;
typedef HcalHitFilter<HcalOuter> HOHitFilter;

#endif
