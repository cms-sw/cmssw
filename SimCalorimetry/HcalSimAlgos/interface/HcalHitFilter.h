#ifndef HcalSimAlgos_HcalHitFilter_h
#define HcalSimAlgos_HcalHitFilter_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitFilter.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


class HcalHitFilter : public CaloVHitFilter 
{
public:
  HcalHitFilter() {}
  virtual ~HcalHitFilter() {}

  void setSubdets(const std::vector<HcalSubdetector> subdets);
  void setDetIds(const std::vector<DetId> & detIds);

  virtual bool accepts(const PCaloHit & hit) const;

protected:
  std::vector<HcalSubdetector> theSubdets;
  // empty DetIds will always be accepted
  std::vector<DetId> theDetIds;
};

#endif

