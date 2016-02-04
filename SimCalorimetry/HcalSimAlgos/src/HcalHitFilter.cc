#include "SimCalorimetry/HcalSimAlgos/interface/HcalHitFilter.h"


HcalHitFilter::HcalHitFilter(HcalSubdetector subdet) 
: theSubdet(subdet) 
{
}


void HcalHitFilter::setDetIds(const std::vector<DetId> & detIds) 
{
  theDetIds = detIds;
}


bool HcalHitFilter::accepts(const PCaloHit & hit) const {
  bool result = false;
  HcalDetId hcalDetId(hit.id());
  if(hcalDetId.subdet() == theSubdet)
  {
    if(theDetIds.empty() || std::find(theDetIds.begin(), theDetIds.end(), DetId(hit.id())) != theDetIds.end())
    {
      result = true;
    }
  }
  return result;
}

