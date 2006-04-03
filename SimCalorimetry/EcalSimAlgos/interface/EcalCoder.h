
#ifndef EcalSimAlgos_EcalCoder_h
#define EcalSimAlgos_EcalCoder_h 1

class EcalMGPASample;
class EcalPedestals;
class EBDataFrame;
class EEDataFrame;
class CaloSamples;
class DetId;
#include<vector>

/* \class EEDigitizerTraits
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 *
 */
class EcalCoder
{
 public:
  /// number of available bits
  enum {NBITS = 12};
  // 2^12 -1
  /// adc max range
  enum {MAXADC = 4095}; 
  /// number of electronic gains
  enum {NGAINS = 3};

  /// ctor
  EcalCoder(bool addNoise) ;
  /// dtor
  virtual ~EcalCoder() {}

  /// can be fetched every event from the EventSetup
  void setPedestals(const EcalPedestals * pedestals) {thePedestals = pedestals;}

  /// from EBDataFrame to CaloSamples
  virtual void digitalToAnalog(const EBDataFrame& df, CaloSamples& lf) const;
  /// from EEDataFrame to CaloSamples
  virtual void digitalToAnalog(const EEDataFrame& df, CaloSamples& lf) const;
  /// from CaloSamples to EBDataFrame
  virtual void analogToDigital(const CaloSamples& clf, EBDataFrame& df) const;
  /// from CaloSamples to EEDataFrame
  virtual void analogToDigital(const CaloSamples& clf, EEDataFrame& df) const;
 
  ///  anything that needs to be done once per event
  void newEvent() {}

 private:

  /// limit on the energy scale due to the electronics range
  double fullScaleEnergy (const DetId & ) const ;

  /// produce the pulse-shape
  std::vector<EcalMGPASample> encode(const CaloSamples& timeframe) const;

  double decode(const EcalMGPASample & sample, const DetId & detId) const;

  /// not yet implemented
  void noisify(float * values, int size) const;

  /// look for the right pedestal according to the electronics gain
  void findPedestal(const DetId & detId, int gainId, 
                    double & pedestal, double & width) const;
  
  /// the pedestals
  const EcalPedestals * thePedestals;
  /// the electronics gains
  double theGains[NGAINS];
  /// the electronics gains errors
  double theGainErrors[NGAINS];
  /// max attainable energy in the ecal barrel
  double m_maxEneEB ;
  /// max attainable energy in the ecal endcap
  double m_maxEneEE ;
  /// whether add noise to the pedestals and the gains
  bool addNoise_;
};


#endif
