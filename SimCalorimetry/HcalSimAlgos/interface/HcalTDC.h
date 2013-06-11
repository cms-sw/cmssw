#ifndef HcalSimAlgos_HcalTDC_h
#define HcalSimAlgos_HcalTDC_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDCParameters.h"
#include "CLHEP/Random/RandGaussQ.h"
class HcalDbService;

class HcalTDC {

public:
  HcalTDC(unsigned int thresholdDAC = 12);
  ~HcalTDC();

  /// adds timing information to the digi
  void timing(const CaloSamples & lf, HcalUpgradeDataFrame & digi) const;

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService * service);
  void setRandomEngine(CLHEP::HepRandomEngine & engine);

  void setThresholdDAC(unsigned int DAC) { theDAC = DAC; }
  unsigned int getThresholdDAC() { return theDAC; }

private:
  double getThreshold(const HcalGenericDetId & detId) const;
  HcalTDCParameters theTDCParameters;
  const HcalDbService * theDbService;

  unsigned int theDAC;

  mutable CLHEP::RandGaussQ * theRandGaussQ;
};

#endif

