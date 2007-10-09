#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalNominalCoder.h"



CastorCoderFactory::CastorCoderFactory(CoderType coderType) 
: theCoderType(coderType),
  theDbService(0)
{
}


std::auto_ptr<HcalCoder> CastorCoderFactory::coder(const DetId & id) const {
  HcalCoder * result = 0;
  if(theCoderType == DB) {
    assert(theDbService != 0);
    HcalGenericDetId hcalGenDetId(id);
    const HcalQIECoder * qieCoder = theDbService->getHcalCoder(hcalGenDetId );
    const HcalQIEShape * qieShape = theDbService->getHcalShape();
    result = new HcalCoderDb(*qieCoder, *qieShape);
  }

  else {
    result = new HcalNominalCoder();
  }
  return std::auto_ptr<HcalCoder>(result);
}

