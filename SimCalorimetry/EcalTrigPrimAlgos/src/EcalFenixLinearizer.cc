#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>

EcalFenixLinearizer::EcalFenixLinearizer()
{
    //UB FIXME : should come from a database!!!
    edm::FileInPath fileInPath("SimCalorimetry/EcalTrigPrimProducers/data/FenixStripPedMult.txt");
    FILE * fin=fopen(fileInPath.fullPath().c_str(), "r");
    //    if (!fin) throw an exception
    unsigned int tow, str, strXtal, gain, ped, mult,shift;
    for(int i=0; i<5100; i++)
      {
	//	fscanf(fin,"%x %d %d %d %x %x %x\n",&tow, &str, &strXtal, &gain, &ped, &bid1, &bid2);
	fscanf(fin,"%u %u %u %u %x %x %u\n",&tow, &str, &strXtal, &gain, &ped, &mult, &shift);
	if(gain==12) gain=1;
	else if(gain==6) gain=2;
	else if(gain==1) gain=3;
	baseLine_[tow-1][str-1][strXtal-1][gain-1]=ped;
	multLine_[tow-1][str-1][strXtal-1][gain-1]=mult;
	shiftLine_[tow-1][str-1][strXtal-1][gain-1]=shift;


      }

    fclose(fin);
}

EcalFenixLinearizer::~EcalFenixLinearizer(){
}


void EcalFenixLinearizer::process(const EBDataFrame &df,int stripnr, int townr, EBDataFrame *dfout)
{


//We know a tower numbering is:
// S1 S2 S3 S4 S5
//
// 4  5  14 15 24
// 3  6  13 16 23
// 2  7  12 17 22
// 1  8  11 18 21
// 0  9  10 19 20

  int crystalNumberInStrip=((df.id()).ic()-1)%numberOfCrystalsInStrip;
  if ((df.id()).ieta()<0) crystalNumberInStrip=numberOfCrystalsInStrip - crystalNumberInStrip - 1;

  dfout->setSize(SIZEMAX);
  for (int i=0;i<df.size();i++) {
    setInput(df[i],stripnr, townr,crystalNumberInStrip);
    dfout->setSample(i,EcalMGPASample(process(),gainID_)); 
  }
  return;
}	 

int EcalFenixLinearizer::process()
{
  int output=(uncorrectedSample_-base_); //Substract base
  if(output<0) return 0;
  output=(output*mult_)>>shift_;        //Apply multiplicative factor
  if(output>0X3FFFF)output=0X3FFFF;         //Saturation if too high
  return output;
}

int EcalFenixLinearizer::setInput(EcalMGPASample RawSam, int stripNum, int towNum,int XtalNumberInStrip)
{
  strip_=stripNum;
  tow_=towNum-1;
  if(RawSam.raw()>0X3FFF)
    {
      std::cout<<"ERROR IN INPUT SAMPLE OF FENIX LINEARIZER"<<std::endl;
      return -1;
    }
  uncorrectedSample_=RawSam.adc(); //uncorrectedSample_ is coded in the 12 LSB
  gainID_=RawSam.gainId();       //uncorrectedSample_ is coded in the 2 next bits!
  gainID_--;       //To have gainID_ in the range [0; 2] and not [1;3]!! 

  base_=getBase(XtalNumberInStrip);
  mult_=getMult(XtalNumberInStrip);
  shift_=getShift(XtalNumberInStrip);

  return 1;
}

int EcalFenixLinearizer::getBase(int XtalNumberInStrip) const
{
  

  return baseLine_[tow_][strip_][XtalNumberInStrip][gainID_];
  
}
int EcalFenixLinearizer::getMult(int XtalNumberInStrip) const
{
  return multLine_[tow_][strip_][XtalNumberInStrip][gainID_];
}

int EcalFenixLinearizer::getShift(int XtalNumberInStrip) const
{
 
  return shiftLine_[tow_][strip_][XtalNumberInStrip][gainID_]+2;
}
