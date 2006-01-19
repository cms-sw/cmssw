#ifndef HcalElectronicsSim_h
#define HcalElectronicsSim_h
  
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

class HBHEDataFrame;
class HODataFrame;
class HFDataFrame;

namespace cms { 

  class HcalAmplifier;
  class HcalCoderFactory;
  
  class HcalElectronicsSim {
  public:
    HcalElectronicsSim(HcalAmplifier * amplifier, 
                       const HcalCoderFactory * coderFactory);
    /// doesn't delete pointers
    ~HcalElectronicsSim() {}

    void analogToDigital(CaloSamples & linearFrame, HBHEDataFrame & result, bool addNoise);
    void analogToDigital(CaloSamples & linearFrame, HODataFrame & result, bool addNoise);
    void analogToDigital(CaloSamples & linearFrame, HFDataFrame & result, bool addNoise);

    /// Things that need to be initialized every event
    void newEvent();

  private:
    template<class Digi> void convert(CaloSamples & frame, Digi & result, bool addNoise);

    HcalAmplifier * theAmplifier;
    const HcalCoderFactory * theCoderFactory;

    int theStartingCapId;
  };
}
  
#endif
  
