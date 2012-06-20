#include "SimCalorimetry/HcalSimAlgos/interface/HcalCoderFactory.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderUpgrade.h"


HcalCoderFactory::HcalCoderFactory(CoderType coderType) 
: theCoderType(coderType),
  theDbService(0)
{
}

void HcalCoderFactory::setDbService(const HcalDbService * service) 
{
  theDbService = service;
}

std::auto_ptr<HcalCoder> HcalCoderFactory::coder(const DetId & id) const {
  HcalCoder * result = 0;
  if(theCoderType == DB) {
    assert(theDbService != 0);
    HcalGenericDetId hcalGenDetId(id);
    const HcalQIECoder * qieCoder = theDbService->getHcalCoder(hcalGenDetId );
    const HcalQIEShape * qieShape = theDbService->getHcalShape();
    result = new HcalCoderDb(*qieCoder, *qieShape);
  } else if (theCoderType == UPGRADE) {
    assert(theDbService != 0);
    HcalGenericDetId hcalGenDetId(id);
    const HcalQIECoder * qieCoder = theDbService->getHcalCoder(hcalGenDetId);
    const HcalQIEShape * qieShape = theDbService->getHcalShape();
    result = new HcalCoderUpgrade(*qieCoder, *qieShape);
  } else {
    result = new HcalNominalCoder();
  }
  return std::auto_ptr<HcalCoder>(result);
}

