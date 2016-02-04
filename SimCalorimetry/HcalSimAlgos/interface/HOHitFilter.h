#ifndef HcalSimAlgos_HOHitFilter_h
#define HcalSimAlgos_HOHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"



class HOHitFilter : public CaloVHitFilter {
  virtual bool accepts(const PCaloHit & hit) const {
    HcalDetId hcalDetId(hit.id());
    return (hcalDetId.subdet() == HcalOuter);
  }
};

#endif

