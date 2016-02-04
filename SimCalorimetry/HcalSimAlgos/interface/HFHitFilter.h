#ifndef HcalSimAlgos_HFHitFilter_h
#define HcalSimAlgos_HFHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <iostream>

class HFHitFilter : public CaloVHitFilter {
  public:
  HFHitFilter(bool doHFWindow): doHFWindow_(doHFWindow) {}

  virtual bool accepts(const PCaloHit & hit) const {
    HcalDetId hcalDetId(hit.id());
    bool ok1 = hcalDetId.subdet() == HcalForward;
    // might not want depth=1
    //if(ok1) std::cout << " HF " << hcalDetId.subdet() << " " << doHFWindow_ << " DEPTH " << hit.depth() << std::endl;
    bool ok2 = doHFWindow_ || hit.depth()==0;
    return ok1 && ok2;
  }
private:
  bool doHFWindow_;
};

#endif

