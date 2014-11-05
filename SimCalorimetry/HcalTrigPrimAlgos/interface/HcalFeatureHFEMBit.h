#ifndef SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureHFEMBit_h_included
#define SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureHFEMBit_h_included 1


#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFeatureBit.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

class HcalFeatureHFEMBit : public HcalFeatureBit {
public:    
    HcalFeatureHFEMBit(double ShortMinE, double LongMinE, double ShortLongCutSlope, double ShortLongCutOffset, const HcalDbService& conditions);
    ~HcalFeatureHFEMBit();
    virtual bool fineGrainbit(int ADCShort, HcalDetId Sid, int CapIdS, int ADCLong, HcalDetId Lid, int CapIdL) const;//cuts based on energy 
    //depoisted in the long and short fibers
private:
    double ShortMinE_, LongMinE_, ShortLongCutSlope_, ShortLongCutOffset_;
    const HcalDbService& conditions_;
};
#endif
