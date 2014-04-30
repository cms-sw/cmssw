#ifndef SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureBit_h_included
#define SimCalorimetry_HcalTPGAlgos_interface_HcalFeatureBit_h_included 1

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalFeatureBit {
	public:
		HcalFeatureBit(){}
		virtual ~HcalFeatureBit(){} // needs to be virtual to avoid memory leaks
		virtual bool fineGrainbit(int ADCShort, HcalDetId Sid, int ADCLong, HcalDetId Lid ){return false;}
		
};
#endif

