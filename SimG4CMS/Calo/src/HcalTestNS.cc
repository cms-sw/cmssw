#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "SimG4CMS/Calo/interface/HcalNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HcalTestNS.h"

//#define EDM_ML_DEBUG

HcalTestNS::HcalTestNS(const edm::EventSetup* iSetup) {

  edm::ESHandle<HcalDDDRecConstants>    hdc;
  iSetup->get<HcalRecNumberingRecord>().get(hdc);
  if (hdc.isValid()) {
    hcons_ = hdc.product();
  } else {
    edm::LogError("HcalSim") << "HcalTestNS : Cannot find HcalDDDRecConstant";
    hcons_ = nullptr;
  }
}

HcalTestNS::~HcalTestNS() {}

bool HcalTestNS::compare(HcalNumberingFromDDD::HcalID const& tmp, 
			 uint32_t const& id) {

  HcalNumberingScheme* scheme = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(false));
  uint32_t id0 = scheme->getUnitID(tmp);
  DetId    hid = HcalHitRelabeller::relabel(id0,hcons_);
  bool     ok  =  (id == hid.rawId());
#ifdef EDM_ML_DEBUG
  std::cout << "Det ID from HCalSD " << HcalDetId(id) << " " << std::hex << id
	    << std::dec << " from relabller " << HcalDetId(hid) << " " 
	    << std::hex << hid.rawId() << std::dec;
  if (!ok) std::cout << " **** ERROR ****" << std::endl;
  else     std::cout << " OK " << std::endl;
#endif
  return ok;
}
