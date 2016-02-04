#ifndef HcalSimAlgos_HcalHitFilter_h
#define HcalSimAlgos_HcalHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


class HcalHitFilter : public CaloVHitFilter 
{
public:
  explicit HcalHitFilter(HcalSubdetector subdet);
  virtual ~HcalHitFilter() {}

  void setDetIds(const std::vector<DetId> & detIds);

  virtual bool accepts(const PCaloHit & hit) const;

private:
  HcalSubdetector theSubdet;
  // empty DetIds will always be accepted
  std::vector<DetId> theDetIds;
};

#endif

