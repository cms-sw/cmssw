#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorNominalCoder.h"
#include "SimCalorimetry/CastorSim/src/CastorCoderFactory.h"
#include <cassert>

CastorCoderFactory::CastorCoderFactory(CoderType coderType) : theCoderType(coderType), theDbService(nullptr) {}

std::unique_ptr<CastorCoder> CastorCoderFactory::coder(const DetId &id) const {
  CastorCoder *result = nullptr;
  if (theCoderType == DB) {
    assert(theDbService != nullptr);
    HcalGenericDetId hcalGenDetId(id);
    const CastorQIECoder *qieCoder = theDbService->getCastorCoder(hcalGenDetId);
    const CastorQIEShape *qieShape = theDbService->getCastorShape();
    result = new CastorCoderDb(*qieCoder, *qieShape);
  }

  else {
    result = new CastorNominalCoder();
  }
  return std::unique_ptr<CastorCoder>(result);
}
