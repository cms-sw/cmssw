#ifndef HcalSimAlgos_ZDCHitFilter_h
#define HcalSimAlgos_ZDCHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

class ZDCHitFilter : public CaloVHitFilter {
  virtual bool accepts(const PCaloHit & hit) const {
    DetId detId(hit.id());
    return (detId.det()==DetId::Calo && detId.subdetId()==HcalZDCDetId::SubdetectorId);
  }
};

#endif

