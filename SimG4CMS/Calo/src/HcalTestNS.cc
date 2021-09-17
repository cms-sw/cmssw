#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNS.h"

//#define EDM_ML_DEBUG

HcalTestNS::HcalTestNS(const HcalDDDRecConstants* hcons) : hcons_(hcons) {}

HcalTestNS::~HcalTestNS() {}

bool HcalTestNS::compare(HcalNumberingFromDDD::HcalID const& tmp, uint32_t const& id) {
  HcalNumberingScheme* scheme = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(false));
  uint32_t id0 = scheme->getUnitID(tmp);
  DetId hid = HcalHitRelabeller::relabel(id0, hcons_);
  bool ok = (id == hid.rawId());
#ifdef EDM_ML_DEBUG
  std::string ck = (ok ? " OK " : " **** ERROR ****");
  edm::LogVerbatim("HcalSim") << "HcalTestNS:: Det ID from HCalSD " << HcalDetId(id) << " " << std::hex << id
                              << std::dec << " from relabller " << HcalDetId(hid) << " " << std::hex << hid.rawId()
                              << std::dec << ck;
#endif
  return ok;
}
