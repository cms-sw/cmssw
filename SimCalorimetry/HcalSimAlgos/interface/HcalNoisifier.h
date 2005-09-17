#ifndef HcalNoisifier_h
#define HcalNoisifier_h
  
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVNoisifier.h"
class HcalDbServiceHandle;

namespace cms {
  class HcalSimCalibrator;
  
  class HcalNoisifier : public CaloVNoisifier {
  public:
    explicit HcalNoisifier(HcalDbServiceHandle * calibrator);
    /// doesn't delete the pointer
    virtual ~HcalNoisifier(){}
  
    virtual void noisify(CaloSamples & linearFrame) const;
  
  private:
    unsigned theStartingCapId;
    HcalDbServiceHandle * theCalibrationHandle;
  };
} 
#endif
