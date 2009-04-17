#include "SimCalorimetry/HcalZeroSuppressionAlgos/interface/HcalZSAlgoRealistic.h"
#include <iostream>

HcalZSAlgoRealistic::HcalZSAlgoRealistic(HcalZeroSuppressionAlgo::ZSMode mode, int levelHB, int levelHE, int levelHO, int levelHF) : 
  HcalZeroSuppressionAlgo(mode),
  thresholdHB_(levelHB),
  thresholdHE_(levelHE),
  thresholdHO_(levelHO),
  thresholdHF_(levelHF)
{
}

namespace ZSRealistic_impl {

  template <class DIGI> 
  bool keepMe(const DIGI& inp, int threshold) {
    bool keepIt=false;

    //determine the sum of 2 timeslices

    for (int i=0; i< inp.size()-1 && !keepIt; i++) {
      int sum=0;
      for (int j=i; j<(i+2); j++){
	sum+=inp[j].adc();
	//pedsum+=pedave;
      }
      if (sum>=threshold) keepIt=true;
    }
    return keepIt;
  }
}


bool HcalZSAlgoRealistic::shouldKeep(const HBHEDataFrame& digi) const {
  if (digi.id().subdet()==HcalBarrel) return ZSRealistic_impl::keepMe<HBHEDataFrame>(digi,thresholdHB_);
  else return ZSRealistic_impl::keepMe<HBHEDataFrame>(digi,thresholdHE_);
}

bool HcalZSAlgoRealistic::shouldKeep(const HODataFrame& digi) const {
  return ZSRealistic_impl::keepMe<HODataFrame>(digi,thresholdHO_);
}
bool HcalZSAlgoRealistic::shouldKeep(const HFDataFrame& digi) const {
  return ZSRealistic_impl::keepMe<HFDataFrame>(digi,thresholdHF_);
}
