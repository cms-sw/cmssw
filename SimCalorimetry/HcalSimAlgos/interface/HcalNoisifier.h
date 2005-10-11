#ifndef HcalNoisifier_h
#define HcalNoisifier_h
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoisifier.h"
class HcalDbService;

namespace cms {
  class HcalSimCalibrator;
  
  class HcalNoisifier : public CaloVNoisifier {
  public:
    explicit HcalNoisifier(HcalDbService * calibrator);
    /// doesn't delete the pointer
    virtual ~HcalNoisifier(){}
  
    virtual void noisify(CaloSamples & linearFrame) const;
  
  private:
    unsigned theStartingCapId;
    HcalDbService * theCalibrationHandle;
  };
} 
#endif
