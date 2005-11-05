
#ifndef EcalCoder_h
#define EcalCoder_h 1

#include "CLHEP/Random/RandGaussQ.h"

class EcalMGPASample;
class EcalPedestals;
class EBDataFrame;
class EEDataFrame;
class CaloSamples;
class DetId;
#include<vector>

//! Converts CaloDataFrame in CaloTimeSample and vice versa.

class EcalCoder
{
 public:
  enum {NBITS = 12};
  /// 2^12 -1
  enum {MAXINT = 4095}; 
  enum {NGAINS = 3};
  EcalCoder() ;
  virtual ~EcalCoder() {}

  /// can be fetched every event from the EventSetup
  void setPedestals(const EcalPedestals * pedestals) {thePedestals = pedestals;}

  virtual void digitalToAnalog(const EBDataFrame& df, CaloSamples& lf) const;
  virtual void digitalToAnalog(const EEDataFrame& df, CaloSamples& lf) const;
  virtual void analogToDigital(const CaloSamples& clf, EBDataFrame& df, bool addNoise) const;
  virtual void analogToDigital(const CaloSamples& clf, EEDataFrame& df, bool addNoise) const;
 
 private:
  double fullScaleEnergy(const DetId & ) const {return 1600.;}

  std::vector<EcalMGPASample> encode(const CaloSamples& timeframe, bool addNoise) const;

  double decode(const EcalMGPASample & sample, const DetId & detId) const;

  void noisify(float * values, int size) const;

  void findPedestal(const DetId & detId, int gainId, 
                    double & pedestal, double & width) const;
  
  mutable RandGaussQ theRandomGaussian;

  const EcalPedestals * thePedestals;
  double theGains[NGAINS];
  double theGainErrors[NGAINS];
} ;


#endif
