#ifndef EcalElectronicsSim_h
#define EcalElectronicsSim_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace cms {

  class EBDataFrame;
  class EEDataFrame;
  
  /** This class turns a CaloSamples, representing the analog
      signal input to the readout electronics, into a
      digitized dataframe
   */
  
  
  class EcalElectronicsSim {
  public:
    EcalElectronicsSim() {}
  
    void run(const CaloSamples & linearFrame, EBDataFrame & result);
    void run(const CaloSamples & linearFrame, EEDataFrame & result);
  
  private:
    template<class Digi> void convert(const CaloSamples & linearFrame, Digi & result) {
      result.setSize(linearFrame.size());
      DetId id = linearFrame.id();
      for(int tbin = 0; tbin < linearFrame.size(); ++tbin) {
      }
    }
  
  };

}
  
#endif
  
