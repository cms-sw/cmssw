#ifndef HcalSimAlgos_HcalElectronicsSim_h
#define HcalSimAlgos_HcalElectronicsSim_h
  
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

class HBHEDataFrame;
class HODataFrame;
class HFDataFrame;

class HcalAmplifier;
class HcalCoderFactory;

class HcalElectronicsSim {
public:
  HcalElectronicsSim(HcalAmplifier * amplifier, 
                     const HcalCoderFactory * coderFactory);
  /// doesn't delete pointers
  ~HcalElectronicsSim() {}

  void analogToDigital(CaloSamples & linearFrame, HBHEDataFrame & result);
  void analogToDigital(CaloSamples & linearFrame, HODataFrame & result);
  void analogToDigital(CaloSamples & linearFrame, HFDataFrame & result);

  /// Things that need to be initialized every event
  void newEvent();

private:
  template<class Digi> void convert(CaloSamples & frame, Digi & result);

  HcalAmplifier * theAmplifier;
  const HcalCoderFactory * theCoderFactory;

  int theStartingCapId;
};

  
#endif
  
