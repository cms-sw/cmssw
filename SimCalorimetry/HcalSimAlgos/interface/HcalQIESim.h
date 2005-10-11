#ifndef HcalQIESim_h
#define HcalQIESim_h
  
class DetId;
class HcalQIESample;

namespace cms {
  
  class HcalQIESim {
  public:
    HcalQIESim();
  
    HcalQIESample makeSample(const DetId & id, int timeBin, double signal);
  
  private:
    unsigned theStartingCapId;
  };
} 
#endif
