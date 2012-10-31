#include "HcalZSAlgoEnergy.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include <iostream>

HcalZSAlgoEnergy::HcalZSAlgoEnergy(bool mp,int level, int start, int samples, bool twosided) : 
  HcalZeroSuppressionAlgo(mp),
  threshold_(level),
  firstsample_(start),
  samplecount_(samples),
  twosided_(twosided)
{
}

namespace ZSEnergy_impl {
  
  template <class DIGI> 
  bool keepMe(const HcalDbService& db, const DIGI& inp, int threshold, int firstSample, int samples, bool twosided) {
    bool keepIt=false;
    const HcalQIEShape* shape = db.getHcalShape (); // this one is generic
    const HcalQIECoder* channelCoder = db.getHcalCoder (inp.id());

    // determine average pedestal for channel
    float pedsum=0, pedave=0, offset=0, slope=0;
    const HcalPedestal* ped=db.getPedestal(inp.id());
    for (int j=0; j<4; j++) {
      pedave+=ped->getValue(j)/4.0;
      offset+=channelCoder->charge(*shape,0,j)/4.0;
      slope+=channelCoder->charge(*shape,1,j)/4.0;
    }
    slope-=offset;
    pedave-=offset;
    pedave/=slope;
      
    int sum=0;

    for (int j=0; j<samples && j+firstSample < inp.size(); j++) {
      sum+=inp[j+firstSample].adc();
      pedsum+=pedave;
    }
    //    int presum=sum;
    sum-=(int)(pedsum);

    if (sum>=threshold) keepIt=true;
    else if (sum<=(-threshold) && twosided) keepIt=true;
    /*
      else
     std::cout << inp.id() << " " << sum << ":" << presum << " " << threshold 
      << " " << pedsum << " " << pedave
      << " " << offset << " " << slope 
      << std::endl;
    */
    return keepIt;
  }
}

bool HcalZSAlgoEnergy::shouldKeep(const HBHEDataFrame& digi) const {
  return ZSEnergy_impl::keepMe<HBHEDataFrame>(*db_,digi,threshold_,firstsample_,samplecount_,twosided_);
}
bool HcalZSAlgoEnergy::shouldKeep(const HODataFrame& digi) const {
  return ZSEnergy_impl::keepMe<HODataFrame>(*db_,digi,threshold_,firstsample_,samplecount_,twosided_);
}
bool HcalZSAlgoEnergy::shouldKeep(const HFDataFrame& digi) const {
  return ZSEnergy_impl::keepMe<HFDataFrame>(*db_,digi,threshold_,firstsample_,samplecount_,twosided_);
}

void HcalZSAlgoEnergy::prepare(const HcalDbService* db) { db_=db; }
void HcalZSAlgoEnergy::done() { db_=0; }
