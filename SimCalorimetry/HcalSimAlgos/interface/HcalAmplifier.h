#ifndef HcalAmplifier_h
#define HcalAmplifier_h
  
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
class HcalDbService;

namespace cms {
  
  class HcalAmplifier {
  public:
    HcalAmplifier(bool addNoise);
    /// doesn't delete the pointer
    virtual ~HcalAmplifier(){}
  
    /// the Producer will probably update this every event
    void setDbService(const HcalDbService * service) {
      theDbService = service;
     }

    virtual void amplify(CaloSamples & linearFrame) const;
  
    void setStartingCapId(int capId) {theStartingCapId = capId;}

  private:
    const HcalDbService * theDbService;
    unsigned theStartingCapId;
    bool addNoise_;
  };
} 
#endif
