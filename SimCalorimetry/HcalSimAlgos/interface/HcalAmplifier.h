#ifndef HcalSimAlgos_HcalAmplifier_h
#define HcalSimAlgos_HcalAmplifier_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
class CaloVSimParameterMap;
class CaloVNoiseSignalGenerator;
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

class HcalDbService;
class HPDIonFeedbackSim;
class HcalTimeSlewSim;
class HcalTimeSlew;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalAmplifier {
public:
  HcalAmplifier(const CaloVSimParameterMap* parameters, bool addNoise, bool PreMix1, bool PreMix2);

  virtual ~HcalAmplifier() {}

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService* service);
  void setIonFeedbackSim(HPDIonFeedbackSim* feedbackSim) { theIonFeedbackSim = feedbackSim; }

  /// if it's set, the amplifier will only use it to check
  /// if it has already added noise
  void setNoiseSignalGenerator(const CaloVNoiseSignalGenerator* noiseSignalGenerator) {
    theNoiseSignalGenerator = noiseSignalGenerator;
  }
  void setTimeSlewSim(HcalTimeSlewSim* timeSlewSim) { theTimeSlewSim = timeSlewSim; }

  const HcalTimeSlew* theTimeSlew = nullptr;
  void setTimeSlew(const HcalTimeSlew* timeSlew) { theTimeSlew = timeSlew; }

  virtual void amplify(CaloSamples& linearFrame, CLHEP::HepRandomEngine*) const;

  void setStartingCapId(int capId) { theStartingCapId = capId; }

private:
  void pe2fC(CaloSamples& frame) const;
  void applyQIEdelay(CaloSamples& frame, int delayQIE) const;
  void addPedestals(CaloSamples& frame, CLHEP::HepRandomEngine*) const;
  void makeNoise(HcalGenericDetId::HcalGenericSubdetector hcalSubDet,
                 const HcalCalibrationWidths& width,
                 int fFrames,
                 double* fGauss,
                 double* fNoise) const;

  const HcalDbService* theDbService;
  const CaloVSimParameterMap* theParameterMap;
  const CaloVNoiseSignalGenerator* theNoiseSignalGenerator;
  HPDIonFeedbackSim* theIonFeedbackSim;
  HcalTimeSlewSim* theTimeSlewSim;
  unsigned theStartingCapId;
  bool addNoise_;
  bool preMixDigi_;
  bool preMixAdd_;
};

#endif
