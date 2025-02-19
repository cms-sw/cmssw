#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorNominalCoder.h"



CastorCoderFactory::CastorCoderFactory(CoderType coderType) 
: theCoderType(coderType),
  theDbService(0)
{
}


std::auto_ptr<CastorCoder> CastorCoderFactory::coder(const DetId & id) const {
  CastorCoder * result = 0;
  if(theCoderType == DB) {
    assert(theDbService != 0);
    HcalGenericDetId hcalGenDetId(id);
    const CastorQIECoder * qieCoder = theDbService->getCastorCoder(hcalGenDetId );
    const CastorQIEShape * qieShape = theDbService->getCastorShape();
    result = new CastorCoderDb(*qieCoder, *qieShape);
  }

  else {
    result = new CastorNominalCoder();
  }
  return std::auto_ptr<CastorCoder>(result);
}

