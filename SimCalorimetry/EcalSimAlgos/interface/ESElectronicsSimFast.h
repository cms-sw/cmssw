#ifndef EcalSimAlgos_ESElectronicsSimFast_h
#define EcalSimAlgos_ESElectronicsSimFast_h 1

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include "CLHEP/Random/RandGeneral.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"

#include <vector>
#include "TFile.h"

class ESElectronicsSimFast
{
 public:
  
  enum {MAXADC = 4095};
  enum {MINADC = 0};
  
  ESElectronicsSimFast (bool addNoise, int gain, ESPedestals peds, ESIntercalibConstants mips, double MIPToGeV);

  void setGain (const int gain) { gain_ = gain; } 
  void setPedestals(const ESPedestals& peds) { peds_ = peds; }
  void setMIPs(const ESIntercalibConstants& mips) { mips_ = mips; }
  void setMIPToGeV (const double MIPToGeV) { MIPToGeV_ = MIPToGeV; }

  virtual void analogToDigital(const CaloSamples& cs, ESDataFrame& df, bool wasEmpty, CLHEP::RandGeneral *histoDistribution, double hInf, double hSup, double hBin) const;
  
  void digitalToAnalog(const ESDataFrame& df, CaloSamples& cs) const; 

  ///  anything that needs to be done once per event
  void newEvent() {}

  private :

    bool addNoise_;
    int gain_;
    ESPedestals peds_;
    ESIntercalibConstants mips_;
    double MIPToGeV_;

    std::vector<ESSample> standEncode(const CaloSamples& timeframe) const;
    std::vector<ESSample> fastEncode(const CaloSamples& timeframe, CLHEP::RandGeneral *histoDistribution, double hInf, double hSup, double hBin) const;

    double decode(const ESSample & sample, const DetId & detId) const;

} ;


#endif
