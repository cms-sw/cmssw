#include "SimCalorimetry/HcalZeroSuppressionAlgos/interface/HcalZSAlgoEnergy.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"

HcalZSAlgoEnergy::HcalZSAlgoEnergy(int level, int start, int samples, bool twosided) : 
  threshold_(level),
  firstsample_(start),
  samplecount_(samples),
  twosided_(twosided)
{
}

namespace ZSEnergy_impl {
  
  template <class DIGI> 
  void suppress(const HcalDbService& db, const DIGI& inp, DIGI& outp, int threshold, int firstSample, int samples, bool twosided) {
    typename DIGI::const_iterator i;

    for (i=inp.begin(); i!=inp.end(); i++) {
      // determine average pedestal for channel
      float pedsum=0, pedave=0;
      const HcalPedestal* ped=db.getPedestal(i->id());
      for (int j=0; j<4; j++)
	pedave+=ped->getValue(j)/4.0;

      int sum=0;

      for (int j=0; j<samples && j+firstSample < i->size(); j++) {
	sum+=(*i)[j+firstSample].adc();
	pedsum+=pedave;
      }
      sum-=(int)(pedsum+0.5);

      if (sum>=threshold)
	outp.push_back(*i);
      else if (sum<=-threshold && twosided)
	outp.push_back(*i);
    }
  }
}


void HcalZSAlgoEnergy::suppress(const HcalDbService& db, const HBHEDigiCollection& inp, HBHEDigiCollection& outp) const {
  ZSEnergy_impl::suppress<HBHEDigiCollection>(db,inp,outp,threshold_,firstsample_,samplecount_,twosided_);
}

void HcalZSAlgoEnergy::suppress(const HcalDbService& db, const HODigiCollection& inp, HODigiCollection& outp) const {
  ZSEnergy_impl::suppress<HODigiCollection>(db,inp,outp,threshold_,firstsample_,samplecount_,twosided_);
}

void HcalZSAlgoEnergy::suppress(const HcalDbService& db, const HFDigiCollection& inp, HFDigiCollection& outp) const {
  ZSEnergy_impl::suppress<HFDigiCollection>(db,inp,outp,threshold_,firstsample_,samplecount_,twosided_);
}
