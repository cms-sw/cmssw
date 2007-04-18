#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/DBInterface.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
#include <fstream>
using namespace std;
EcalFenixLinearizer::EcalFenixLinearizer(DBInterface * db)
  : db_(db)
{
}

EcalFenixLinearizer::~EcalFenixLinearizer(){
}

std::vector<int> EcalFenixLinearizer::process(const EBDataFrame &df)
{
//We know a tower numbering is:
// S1 S2 S3 S4 S5
//
// 4  5  14 15 24
// 3  6  13 16 23
// 2  7  12 17 22
// 1  8  11 18 21
// 0  9  10 19 20

  std::vector<int> output_percry;
  // cout<<"dfsize input lin= "<<df.size()<<endl;
  for (int i=0;i<df.size();i++) {
    setInput(df[i]);
    output_percry.push_back(process());
  }

  return(output_percry);
}	 

int EcalFenixLinearizer::process()
{
  int output=(uncorrectedSample_-base_); //Substract base
  if(output<0) return 0;
  output=(output*mult_)>>(shift_+2); //Apply multiplicative factor
  if(output>0X3FFFF)output=0X3FFFF;         //Saturation if too high
  return output;
}

int EcalFenixLinearizer::setInput(EcalMGPASample RawSam)
{
  if(RawSam.raw()>0X3FFF)
    {
      std::cout<<"ERROR IN INPUT SAMPLE OF FENIX LINEARIZER"<<std::endl;
      return -1;
    }
  uncorrectedSample_=RawSam.adc(); //uncorrectedSample_ is coded in the 12 LSB
  gainID_=RawSam.gainId();       //uncorrectedSample_ is coded in the 2 next bits!
  if (gainID_==0)    gainID_=3;
  gainID_ -- ; 
  base_ = params_[3*gainID_] ;
  mult_ = params_[3*gainID_+1] ;
  shift_ = params_[3*gainID_+2] ;
if (uncorrectedSample_==2568)  std::cout<<"base= "<<base_<<" mult= "<<mult_<<" shift= "<<shift_<<std::endl;
  return 1;
}

void EcalFenixLinearizer::setParameters(int SM, int towNum, int stripNum,int XtalNumberInStrip)
{
  params_ = db_->getXtalParameters(SM, towNum, stripNum, XtalNumberInStrip) ;
}
