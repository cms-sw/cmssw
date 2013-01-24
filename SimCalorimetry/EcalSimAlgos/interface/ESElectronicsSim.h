#ifndef EcalSimAlgos_ESElectronicsSim_h
#define EcalSimAlgos_ESElectronicsSim_h 1

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"

#include<vector>

class ESElectronicsSim
{
 public:

  enum {MAXADC = 4095};
  enum {MINADC = 0};

  ESElectronicsSim (bool addNoise);

  void setGain (const int gain) { gain_ = gain; }
  void setPedestals(const ESPedestals* peds) { peds_ = peds; }
  void setMIPs(const ESIntercalibConstants* mips) { mips_ = mips; }
  void setMIPToGeV (const double MIPToGeV) { MIPToGeV_ = MIPToGeV; }

  virtual void analogToDigital(const CaloSamples& cs, ESDataFrame& df) const;
  virtual void digitalToAnalog(const ESDataFrame& df, CaloSamples& cs) const;

  ///  anything that needs to be done once per event
  void newEvent() {}

  private :

    bool addNoise_;
    int gain_;
    const ESPedestals *peds_;
    const ESIntercalibConstants *mips_;
    double MIPToGeV_;

    std::vector<ESSample> encode(const CaloSamples& timeframe) const;
    double decode(const ESSample & sample, const DetId & detId) const;

} ;


#endif
