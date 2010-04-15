#ifndef HcalSimAlgos_HcalAmplifier_h
#define HcalSimAlgos_HcalAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
class CaloVSimParameterMap;
class CaloVNoiseSignalGenerator;
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "CLHEP/Random/RandGaussQ.h"

class HcalDbService;
class HPDIonFeedbackSim;

class HcalAmplifier {
public:
  HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise);
  virtual ~HcalAmplifier(){ delete theRandGaussQ; }

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService * service);
  void setRandomEngine(CLHEP::HepRandomEngine & engine);
  void setIonFeedbackSim(HPDIonFeedbackSim * feedbackSim) {theIonFeedbackSim = feedbackSim;}

  /// if it's set, the amplifier will only use it to check
  /// if it has already added noise
  void setNoiseSignalGenerator(const CaloVNoiseSignalGenerator * noiseSignalGenerator) {
    theNoiseSignalGenerator = noiseSignalGenerator;
  }

  virtual void amplify(CaloSamples & linearFrame) const;

  void setStartingCapId(int capId) {theStartingCapId = capId;}

private:

  void pe2fC(CaloSamples & frame) const;
  void addPedestals(CaloSamples & frame) const;
  void makeNoise (const HcalCalibrationWidths& width, int fFrames, double* fGauss, double* fNoise) const;

  const HcalDbService * theDbService;
  CLHEP::RandGaussQ * theRandGaussQ;
  const CaloVSimParameterMap * theParameterMap;
  const CaloVNoiseSignalGenerator * theNoiseSignalGenerator;
  HPDIonFeedbackSim * theIonFeedbackSim;
  unsigned theStartingCapId;
  bool addNoise_;
};

#endif
