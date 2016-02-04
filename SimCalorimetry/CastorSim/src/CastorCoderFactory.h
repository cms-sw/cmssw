#ifndef CastorSim_CastorCoderFactory_h
#define CastorSim_CastorCoderFactory_h

#include <memory>
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorCoder.h"

class CastorCoderFactory
{
public:
  enum CoderType {DB, NOMINAL};

  CastorCoderFactory(CoderType coderType);

  void setDbService(const CastorDbService * service) {theDbService = service;}

  /// user gets control of the pointer
  std::auto_ptr<CastorCoder> coder(const DetId & detId) const;

private:

  CoderType theCoderType;
  const CastorDbService * theDbService;
};

#endif

