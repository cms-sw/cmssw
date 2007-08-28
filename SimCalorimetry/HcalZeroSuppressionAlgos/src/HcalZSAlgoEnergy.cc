#include "SimCalorimetry/HcalZeroSuppressionAlgos/interface/HcalZSAlgoEnergy.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include <iostream>

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
    const HcalQIEShape* shape = db.getHcalShape (); // this one is generic

    for (i=inp.begin(); i!=inp.end(); i++) {
      const HcalQIECoder* channelCoder = db.getHcalCoder (i->id());

      // determine average pedestal for channel
      float pedsum=0, pedave=0, offset=0, slope=0;
      const HcalPedestal* ped=db.getPedestal(i->id());
      for (int j=0; j<4; j++) {
	pedave+=ped->getValue(j)/4.0;
	offset+=channelCoder->charge(*shape,0,j)/4.0;
	slope+=channelCoder->charge(*shape,1,j)/4.0;
      }
      slope-=offset;
      pedave-=offset;
      pedave/=slope;
      
      int sum=0;

      for (int j=0; j<samples && j+firstSample < i->size(); j++) {
	sum+=(*i)[j+firstSample].adc();
	pedsum+=pedave;
      }
      int presum=sum;
      sum-=(int)(pedsum+0.5);

      if (sum>=threshold)
	outp.push_back(*i);
      else if (sum<=(-threshold) && twosided)
	outp.push_back(*i);
      /*
      else std::cout << i->id() << " " << sum << ":" << presum << " " << threshold 
		     << " " << pedsum << " " << pedave
		     << " " << offset << " " << slope 
		     << std::endl;
      */
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
