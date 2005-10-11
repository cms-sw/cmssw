#ifndef HcalElectronicsSim_h
#define HcalElectronicsSim_h
  
#include "SimCalorimetry/HcalSimAlgos/interface/HcalQIESim.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include<iostream>
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
class HcalCoder;
class HBHEDataFrame;
class HODataFrame;
class HFDataFrame;

namespace cms { 

  class CaloVNoisifier;
  
  class HcalElectronicsSim {
  public:
    HcalElectronicsSim(CaloVNoisifier * noisifier, HcalCoder * coder);
    /// doesn't delete pointers
    ~HcalElectronicsSim() {}
  
    void run(CaloSamples & linearFrame, HBHEDataFrame & result);
    void run(CaloSamples & linearFrame, HODataFrame & result);
    void run(CaloSamples & linearFrame, HFDataFrame & result);
  
  private:
    template<class Digi> void convert(CaloSamples & frame, Digi & result) {
      result.setSize(frame.size());
      theNoisifier->noisify(frame);
      theCoder->fC2adc(frame, result);
    }
  
    CaloVNoisifier * theNoisifier;
    HcalCoder * theCoder;
  };
}
  
#endif
  
