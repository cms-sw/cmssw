#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"

HcalCoderFactory::HcalCoderFactory(CoderType coderType) : theCoderType(coderType), theDbService(nullptr) {}

std::unique_ptr<HcalCoder> HcalCoderFactory::coder(const DetId& id) const {
  HcalCoder* result = nullptr;
  if (theCoderType == DB) {
    assert(theDbService != nullptr);
    HcalGenericDetId hcalGenDetId(id);
    const HcalQIECoder* qieCoder = theDbService->getHcalCoder(hcalGenDetId);
    const HcalQIEShape* qieShape = theDbService->getHcalShape(hcalGenDetId);
    result = new HcalCoderDb(*qieCoder, *qieShape);
  } else {
    result = new HcalNominalCoder();
  }
  return std::unique_ptr<HcalCoder>(result);
}
