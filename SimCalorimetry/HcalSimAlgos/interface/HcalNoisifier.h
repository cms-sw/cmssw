#ifndef HcalNoisifier_h
#define HcalNoisifier_h
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoisifier.h"
class HcalDbService;

namespace cms {
  
  class HcalNoisifier : public CaloVNoisifier {
  public:
    HcalNoisifier();
    /// doesn't delete the pointer
    virtual ~HcalNoisifier(){}
  
    /// the Producer will probably update this every event
    void setDbService(const HcalDbService * service) {
      theDbService = service;
     }

    virtual void noisify(CaloSamples & linearFrame) const;
  
  private:
    unsigned theStartingCapId;
    const HcalDbService * theDbService;
  };
} 
#endif
