#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixTcpFgvbEE.h>
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h"
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include <iostream>

//---------------------------------------------------------------
EcalFenixTcpFgvbEE::EcalFenixTcpFgvbEE(const EcalTPParameters * ecaltpp)
  : ecaltpp_(ecaltpp)
{
}
//---------------------------------------------------------------
EcalFenixTcpFgvbEE::~EcalFenixTcpFgvbEE()
{
}
//---------------------------------------------------------------
void EcalFenixTcpFgvbEE::process(std::vector<std::vector<int> > & bypasslin_out, int nStr,int bitMask,std::vector<int> & output)
{
  std::vector<int> indexLut(output.size());
  
  int lut_fg = (*params_)[1024];//FIXME
  
  for (unsigned int i=0;i<output.size();i++) {
    output[i]=0;
    indexLut[i]=0;
  }
    
  for (unsigned int i=0;i<output.size();i++) {
    for (int istrip=0;istrip<nStr;istrip++) {
      int res = (bypasslin_out[istrip])[i];
      res = (res >>bitMask) & 1;
      indexLut[i]= indexLut[i] | (res << istrip);
    }
    indexLut[i]= indexLut[i] | (nStr << 5);
     
    int mask = 1<<indexLut[i];
    output[i]= lut_fg & mask;
    if (output[i]>0) output[i]=1;
  }
  //  return output;  
  return;
} 

//------------------------------------------------------------------- 

void EcalFenixTcpFgvbEE::setParameters(int sectorNb, int towNum)
{
 params_ = ecaltpp_->getTowerParameters(sectorNb,towNum);
 }

























