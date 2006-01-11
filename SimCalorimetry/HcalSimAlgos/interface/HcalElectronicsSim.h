#ifndef HcalElectronicsSim_h
#define HcalElectronicsSim_h
  
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized data frame
   */
#include "SimCalorimetry/HcalSimAlgos/interface/HcalQIESim.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

class HBHEDataFrame;
class HODataFrame;
class HFDataFrame;

namespace cms { 

  class HcalNoisifier;
  
  class HcalElectronicsSim {
  public:
    HcalElectronicsSim(HcalNoisifier * noisifier);
    /// doesn't delete pointers
    ~HcalElectronicsSim() {}
  
    void analogToDigital(CaloSamples & linearFrame, HBHEDataFrame & result, bool addNoise);
    void analogToDigital(CaloSamples & linearFrame, HODataFrame & result, bool addNoise);
    void analogToDigital(CaloSamples & linearFrame, HFDataFrame & result, bool addNoise);

    /// the Producer will probably update this every event
    void setDbService(const HcalDbService * service);
  
    /// Things that need to be initialized every event
    void newEvent();

  private:
    template<class Digi> void convert(CaloSamples & frame, Digi & result, bool addNoise);
  
    HcalNoisifier * theNoisifier;
    const HcalDbService * theDbService;

    int theStartingCapId;
  };
}
  
#endif
  
