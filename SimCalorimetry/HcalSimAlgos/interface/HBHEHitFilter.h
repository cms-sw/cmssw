#ifndef HcalSimAlgos_HBHEHitFilter_h
#define HcalSimAlgos_HBHEHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"



class HBHEHitFilter : public CaloVHitFilter {
  virtual bool accepts(const PCaloHit & hit) const {
    HcalDetId hcalDetId(hit.id());
    return (hcalDetId.subdet() == HcalBarrel || hcalDetId.subdet() == HcalEndcap);
  }
};

#endif

