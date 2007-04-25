#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h"
#include <iostream>

//----------------------------------------------------------------------------------------
EcalFenixEtTot::EcalFenixEtTot()
{}
//----------------------------------------------------------------------------------------  
EcalFenixEtTot::~EcalFenixEtTot()
{}
//----------------------------------------------------------------------------------------
std::vector<int> EcalFenixEtTot::process(const std::vector<EBDataFrame *> &calodatafr)
{
    std::vector<int> out;
    return out;
}
//----------------------------------------------------------------------------------------
std::vector<int> EcalFenixEtTot::process(std::vector<std::vector <int> >  bypasslinout, int bitMask)
{

std::vector<int> output(SIZEMAX);
  for (int i=0;i<SIZEMAX;i++){
    output[i]= 0;
  }
  //ENDCAP:MODIFS (mask)
  int mask = (1<<bitMask)-1;
  for(unsigned int istrip=0;istrip<bypasslinout.size();istrip++){
    std::vector<int> temp= bypasslinout[istrip];
    for (unsigned int i=0;i<temp.size();i++) {
      output[i]+= temp[i];
      if (output[i]>mask) output[i]= mask;
    }
  }
  return output;
}
//----------------------------------------------------------------------------------------
