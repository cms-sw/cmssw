#ifndef SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureBit_h_included
#define SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureBit_h_included 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalFeatureBit {
	public:
		HcalFeatureBit(){}
		virtual ~HcalFeatureBit(){} //the virutal function is responcible for applying a cut based on a linear relationship of the energy
        //deposited in the short vers long fibers.
		virtual bool fineGrainbit(int ADCShort, HcalDetId Sid, int CapIdS, int ADCLong, HcalDetId Lid, int CapIdL) const {return false;}
		
};
#endif

