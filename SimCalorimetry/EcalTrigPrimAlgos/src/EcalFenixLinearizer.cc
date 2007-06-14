#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>

#include <CondFormats/L1TObjects/interface/EcalTPParameters.h>

#include <DataFormats/EcalDetId/interface/EBDetId.h>
//#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalFenixLinearizer::EcalFenixLinearizer(const EcalTPParameters *ecaltpp, bool famos)
  : ecaltpp_(ecaltpp),famos_(famos)
{
}

EcalFenixLinearizer::~EcalFenixLinearizer(){
}

void EcalFenixLinearizer::setParameters(int SM, int towNum, int stripNum,int XtalNumberInStrip)
{
  params_ = ecaltpp_->getXtalParameters(SM, towNum, stripNum, XtalNumberInStrip,true);
}

int EcalFenixLinearizer::process()
{
  int output=(uncorrectedSample_-base_); //Substract base
  if(output<0) return 0;
  output=(output*mult_)>>(shift_+2);        //Apply multiplicative factor
  if(output>0X3FFFF)output=0X3FFFF;         //Saturation if too high
  return output;
}

int EcalFenixLinearizer::setInput(const EcalMGPASample &RawSam)
{
  if(RawSam.raw()>0X3FFF)
    {
      LogDebug("EcalTPG")<<"ERROR IN INPUT SAMPLE OF FENIX LINEARIZER";
      return -1;
    }
  uncorrectedSample_=RawSam.adc(); //uncorrectedSample_ is coded in the 12 LSB
  gainID_=RawSam.gainId();       //uncorrectedSample_ is coded in the 2 next bits!
  if (gainID_==0)    gainID_=3;
  gainID_ -- ; 
  if (famos_) base_=200; //FIXME by preparing a correct TPG.txt for Famos
  else base_ = (*params_)[3*gainID_] ;
  mult_ = (*params_)[3*gainID_+1] ;
  shift_ = (*params_)[3*gainID_+2] ;
  return 1;
}

