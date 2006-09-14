#ifndef HcalSimAlgos_HcalCoderFactory_h
#define HcalSimAlgos_HcalCoderFactory_h

#include <memory>
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "CalibFormats/CaloTPG/interface/HcalTPGCompressor.h"

class HcalCoderFactory
{
public:
  enum CoderType {DB, NOMINAL};

  HcalCoderFactory(CoderType coderType);

  void setDbService(const HcalDbService * service) {theDbService = service;}
  void setTPGCoder(const HcalTPGCoder* cc) { theTPGcoder=cc; }
  void setCompressionLUTcoder(const HcalTPGCompressor* cc) { theCompressionCoder=cc; }

  /// user gets control of the pointer
  std::auto_ptr<HcalCoder> coder(const DetId & detId) const;

  /// user does not get control of the pointer
  const HcalTPGCoder* TPGcoder() const { return theTPGcoder; }

  /// user does not get control of the pointer
  const HcalTPGCompressor* compressionLUTcoder() const { return theCompressionCoder; }

private:

  CoderType theCoderType;
  const HcalDbService * theDbService;
  const HcalTPGCoder* theTPGcoder;
  const HcalTPGCompressor* theCompressionCoder;
};

#endif

