#ifndef EcalSimAlgos_ESElectronicsSim_h
#define EcalSimAlgos_ESElectronicsSim_h 1

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"

#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class ESElectronicsSim {
public:
  enum { MAXADC = 4095 };
  enum { MINADC = 0 };

  ESElectronicsSim(bool addNoise);
  virtual ~ESElectronicsSim();

  void setGain(const int gain) { gain_ = gain; }
  void setPedestals(const ESPedestals* peds) { peds_ = peds; }
  void setMIPs(const ESIntercalibConstants* mips) { mips_ = mips; }
  void setMIPToGeV(const double MIPToGeV) { MIPToGeV_ = MIPToGeV; }

  virtual void analogToDigital(CLHEP::HepRandomEngine*, const CaloSamples& cs, ESDataFrame& df) const;
  virtual void digitalToAnalog(const ESDataFrame& df, CaloSamples& cs) const;

  ///  anything that needs to be done once per event
  void newEvent(CLHEP::HepRandomEngine*) {}

private:
  bool addNoise_;
  int gain_;
  const ESPedestals* peds_;
  const ESIntercalibConstants* mips_;
  double MIPToGeV_;

  std::vector<ESSample> encode(const CaloSamples& timeframe, CLHEP::HepRandomEngine*) const;
  double decode(const ESSample& sample, const DetId& detId) const;
};

#endif
