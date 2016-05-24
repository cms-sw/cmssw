#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"
#include <algorithm>

void HcalHitFilter::setSubdets(const std::vector<HcalSubdetector> subdets) 
{
  theSubdets = subdets;
  std::sort(theSubdets.begin(),theSubdets.end());
}

void HcalHitFilter::setDetIds(const std::vector<DetId> & detIds) 
{
  theDetIds = detIds;
  std::sort(theDetIds.begin(),theDetIds.end());
}


bool HcalHitFilter::accepts(const PCaloHit & hit) const {
  HcalDetId hcalDetId(hit.id());
  return ( (theSubdets.empty() || std::binary_search(theSubdets.begin(), theSubdets.end(), hcalDetId.subdet()))
        && (theDetIds.empty() || std::binary_search(theDetIds.begin(), theDetIds.end(), DetId(hit.id())))
  );
}

