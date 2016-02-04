#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>

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
void EcalFenixEtTot::process(std::vector<std::vector <int> >  &bypasslinout, int nStr, int bitMask, std::vector<int> & output)
{

  for (unsigned int i=0;i<output.size();i++){
    output[i]= 0;
  }

  int mask = (1<<bitMask)-1;
  for(int istrip=0;istrip<nStr;istrip++){
    for (unsigned int i=0;i<bypasslinout[istrip].size();i++) {
      output[i]+= (bypasslinout[istrip][i] & mask) ; //fix bug inn case of EE: MSB are set for FG, so need to apply mask in summation.
      if (output[i]>mask) output[i]= mask;
    }
  }
  return;
}
//----------------------------------------------------------------------------------------
