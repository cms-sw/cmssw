#ifndef HcalSimAlgos_HcalAmplifier_h
#define HcalSimAlgos_HcalAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
class CaloVSimParameterMap;
class CaloVNoiseSignalGenerator;
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "CLHEP/Random/RandGaussQ.h"

class HcalDbService;

class HcalAmplifier {
public:
  HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise);
  virtual ~HcalAmplifier(){ delete theRandGaussQ; }

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService * service) {
    theDbService = service;
   }

  void setRandomEngine(CLHEP::HepRandomEngine & engine);

  /// if it's set, the amplifier will only use it to check
  /// if it has already added noise
  void setNoiseSignalGenerator(const CaloVNoiseSignalGenerator * noiseSignalGenerator) {
    theNoiseSignalGenerator = noiseSignalGenerator;
  }

  virtual void amplify(CaloSamples & linearFrame) const;

  void setStartingCapId(int capId) {theStartingCapId = capId;}

  void makeNoise (const HcalCalibrationWidths& width, int fFrames, double* fGauss, double* fNoise) const;

private:
  const HcalDbService * theDbService;
  CLHEP::RandGaussQ * theRandGaussQ;
  const CaloVSimParameterMap * theParameterMap;
  const CaloVNoiseSignalGenerator * theNoiseSignalGenerator;
  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
