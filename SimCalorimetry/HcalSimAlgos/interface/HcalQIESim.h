#ifndef HcalQIESim_h
#define HcalQIESim_h
  
namespace cms {
  class HcalQIESample;
  class DetId;
  
  class HcalQIESim {
  public:
    HcalQIESim();
  
    HcalQIESample makeSample(const DetId & id, int timeBin, double signal);
  
  private:
    unsigned theStartingCapId;
  };
} 
#endif
