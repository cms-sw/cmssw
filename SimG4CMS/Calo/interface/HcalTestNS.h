#ifndef SimG4CMS_HcalTestNS_h
#define SimG4CMS_HcalTestNS_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/HcalCommonData/interface/HcalNumberingFromDDD.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

class HcalNumberingScheme;

class HcalTestNS {

public:    

  HcalTestNS(const edm::EventSetup*);
  virtual ~HcalTestNS();

  bool compare(HcalNumberingFromDDD::HcalID const&, uint32_t const&);

private:

  HcalDDDRecConstants* hcons_;
  const HcalNumberingScheme* scheme_;
};

#endif // HcalTestNS_h
