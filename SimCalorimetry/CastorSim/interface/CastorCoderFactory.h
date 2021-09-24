#ifndef CastorSim_CastorCoderFactory_h
#define CastorSim_CastorCoderFactory_h

#include "CalibFormats/CastorObjects/interface/CastorCoder.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include <memory>

class CastorCoderFactory {
public:
  enum CoderType { DB, NOMINAL };

  CastorCoderFactory(CoderType coderType);

  void setDbService(const CastorDbService *service) { theDbService = service; }

  /// user gets control of the pointer
  std::unique_ptr<CastorCoder> coder(const DetId &detId) const;

private:
  CoderType theCoderType;
  const CastorDbService *theDbService;
};

#endif
