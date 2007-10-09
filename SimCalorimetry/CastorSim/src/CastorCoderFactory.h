#ifndef CastorSim_CastorCoderFactory_h
#define CastorSim_CastorCoderFactory_h

#include <memory>
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

class CastorCoderFactory
{
public:
  enum CoderType {DB, NOMINAL};

  CastorCoderFactory(CoderType coderType);

  void setDbService(const HcalDbService * service) {theDbService = service;}

  /// user gets control of the pointer
  std::auto_ptr<HcalCoder> coder(const DetId & detId) const;

private:

  CoderType theCoderType;
  const HcalDbService * theDbService;
};

#endif

