#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h"
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>
#include <iostream>

EcalFenixStripFgvbEE::EcalFenixStripFgvbEE(const EcalTPParameters * ecaltpp)
  : ecaltpp_(ecaltpp)
{
}

EcalFenixStripFgvbEE::~EcalFenixStripFgvbEE(){
}

void EcalFenixStripFgvbEE::process( std::vector<std::vector<int> > &linout ,std::vector<int> & output)
{
  unsigned int maskFgvb[]={1,2,4,8,0x10};

  int threshold_fg = (*params_)[6];
  int lut_fg = (*params_)[7];
  
  std::vector<int> indexLut(output.size());

  for (unsigned int i=0;i<output.size();i++) {
    output[i]=0;
    indexLut[i]=0;
    for (unsigned int ixtal=0;ixtal<linout.size();ixtal++) {
      int adc=linout[ixtal][i];
      int res= (adc>threshold_fg) ? 1 : 0;
      indexLut[i]=indexLut[i] | (res <<ixtal & maskFgvb[ixtal]);
    }
    int mask = 1<<indexLut[i];
    output[i]= lut_fg & mask;
    if (output[i]>0) output[i]=1;
  }
   
  //  return output;
  return;
}  

void EcalFenixStripFgvbEE::setParameters(int sectorNb, int towNum, int stripNr)
{
   params_ = ecaltpp_->getStripParameters(sectorNb,towNum,stripNr);
  // ecaltpp_->getStripParameters(sectorNb,towNum,stripNr,params_);
}
