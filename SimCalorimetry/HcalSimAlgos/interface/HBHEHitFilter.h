#ifndef HBHEHitFilter_h
#define HBHEHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


namespace cms {

class HBHEHitFilter : public CaloVHitFilter {
  virtual bool accepts(const PCaloHit & hit) const {
    HcalDetId hcalDetId(hit.id());
    return (hcalDetId.subdet() == HcalBarrel || hcalDetId.subdet() == HcalEndcap);
  }
};

}
#endif

