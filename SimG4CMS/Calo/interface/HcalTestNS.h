#ifndef SimG4CMS_HcalTestNS_h
#define SimG4CMS_HcalTestNS_h

#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

class HcalTestNS {
public:
  HcalTestNS(const HcalDDDRecConstants*);
  virtual ~HcalTestNS();

  bool compare(HcalNumberingFromDDD::HcalID const&, uint32_t const&);

private:
  const HcalDDDRecConstants* hcons_;
};

#endif  // HcalTestNS_h
