#ifndef HcalSimAlgos_HcalCoderFactory_h
#define HcalSimAlgos_HcalCoderFactory_h

#include <memory>
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"

class HcalCoderFactory {
public:
  enum CoderType { DB, NOMINAL };

  HcalCoderFactory(CoderType coderType);

  void setDbService(const HcalDbService* service) { theDbService = service; }

  /// user gets control of the pointer
  std::unique_ptr<HcalCoder> coder(const DetId& detId) const;

private:
  CoderType theCoderType;
  const HcalDbService* theDbService;
};

#endif
