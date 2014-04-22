#ifndef HcalSimAlgos_HcalAmplifier_h
#define HcalSimAlgos_HcalAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
class CaloVSimParameterMap;
class CaloVNoiseSignalGenerator;
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandFlat.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrices.h"
#include "CondFormats/HcalObjects/interface/HcalCholeskyMatrix.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

class HcalDbService;
class HPDIonFeedbackSim;
class HcalTimeSlewSim;

class HcalAmplifier {
public:
  HcalAmplifier(const CaloVSimParameterMap * parameters, bool addNoise, bool PreMix1, bool PreMix2);
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
  void setTimeSlewSim(HcalTimeSlewSim * timeSlewSim) {
    theTimeSlewSim = timeSlewSim;
  }

  virtual void amplify(CaloSamples & linearFrame) const;

  void setStartingCapId(int capId) {theStartingCapId = capId;}
  void setHBtuningParameter(double tp);
  void setHEtuningParameter(double tp);
  void setHFtuningParameter(double tp);
  void setHOtuningParameter(double tp);
  void setUseOldHB(bool useOld);
  void setUseOldHE(bool useOld);
  void setUseOldHF(bool useOld);
  void setUseOldHO(bool useOld);
  void setCholesky(const HcalCholeskyMatrices * Cholesky) { myCholeskys = Cholesky; }
  void setADCPeds(const HcalPedestals * ADCPeds) { myADCPeds = ADCPeds; }

private:

  void pe2fC(CaloSamples & frame) const;
  void addPedestals(CaloSamples & frame) const;
  void makeNoiseOld (HcalGenericDetId::HcalGenericSubdetector hcalSubDet, const HcalCalibrationWidths& width, int fFrames, double* fGauss, double* fNoise) const;
  void makeNoise (const HcalCholeskyMatrix & thisChanCholesky, int fFrames, double* fGauss, double* fNoise, int m) const;

  const HcalDbService * theDbService;
  CLHEP::RandGaussQ * theRandGaussQ;
  CLHEP::RandFlat * theRandFlat;
  const CaloVSimParameterMap * theParameterMap;
  const CaloVNoiseSignalGenerator * theNoiseSignalGenerator;
  HPDIonFeedbackSim * theIonFeedbackSim;
  HcalTimeSlewSim * theTimeSlewSim;
  unsigned theStartingCapId;
  bool addNoise_;
  bool preMixDigi_;
  bool preMixAdd_;
  bool useOldHB;
  bool useOldHE;
  bool useOldHF;
  bool useOldHO;

  double HB_ff;
  double HE_ff;
  double HF_ff;
  double HO_ff;
  const HcalCholeskyMatrices * myCholeskys;
  const HcalPedestals * myADCPeds;
};

#endif
