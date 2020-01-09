// -*- mode: c++ -*-
#ifndef HcalSimAlgos_HcalTDC_h
#define HcalSimAlgos_HcalTDC_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalTDCParameters.h"
#include "DataFormats/HcalDigi/interface/QIE11DataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalDbService;

namespace CLHEP {
  class HepRandomEngine;
}

class HcalTDC {
public:
  HcalTDC(double threshold_currentTDC = 0.);
  ~HcalTDC();

  /// adds timing information to the digi
  /// template <class Digi>
  void timing(const CaloSamples& lf, QIE11DataFrame& digi) const;

  /// the Producer will probably update this every event
  void setDbService(const HcalDbService* service);

  void setThresholdDAC(unsigned int DAC) { theDAC = DAC; }
  unsigned int getThresholdDAC() { return theDAC; }
  double getThreshold() const { return threshold_currentTDC_; };

private:
  HcalTDCParameters theTDCParameters;
  const HcalDbService* theDbService;

  unsigned int theDAC;
  double threshold_currentTDC_;
  double const lsb;
};

#endif
