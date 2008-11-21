#ifndef HcalSimAlgos_HFHitFilter_h
#define HcalSimAlgos_HFHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


class HFHitFilter : public CaloVHitFilter {
  virtual bool accepts(const PCaloHit & hit) const {
    HcalDetId hcalDetId(hit.id());
    return (hcalDetId.subdet() == HcalForward);
  }
};

#endif

