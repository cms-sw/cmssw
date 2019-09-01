#ifndef CastorSim_CastorHitFilter_h
#define CastorSim_CastorHitFilter_h

#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"

class CastorHitFilter : public CaloVHitFilter {
  bool accepts(const PCaloHit &hit) const override {
    DetId detId(hit.id());
    return (detId.det() == DetId::Calo && detId.subdetId() == HcalCastorDetId::SubdetectorId);
  }
};

#endif
