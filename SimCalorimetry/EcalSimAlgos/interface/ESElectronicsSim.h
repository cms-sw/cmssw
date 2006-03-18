#ifndef EcalSimAlgos_ESElectronicsSim_h
#define EcalSimAlgos_ESElectronicsSim_h 1

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"

#include<vector>

class ESElectronicsSim
{
 public:

  enum {MAXADC = 4095};
  enum {MINADC = 0};

  ESElectronicsSim (bool addNoise, double sigma, int gain, int baseline, double MIPADC, double MIPkeV);

  void setNoiseSigma (const double sigma);
  void setGain (const int gain);
  void setBaseline (const int baseline);
  void setMIPADC (const double MIPADC);
  void setMIPkeV (const double MIPkeV);

  virtual void analogToDigital(const CaloSamples& cs, ESDataFrame& df) const;
  virtual void digitalToAnalog(const ESDataFrame& df, CaloSamples& cs) const;

  ///  anything that needs to be done once per event
  void newEvent() {}

  private :

    bool addNoise_;
    double sigma_;
    int gain_;
    int baseline_;
    double MIPADC_;
    double MIPkeV_;

    std::vector<ESSample> encode(const CaloSamples& timeframe) const;
    double decode(const ESSample & sample, const DetId & detId) const;

} ;


#endif
