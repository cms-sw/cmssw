#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

  EcalFenixStripFormatEB::EcalFenixStripFormatEB(const EcalTPParameters * ecaltpp) 
    : ecaltpp_(ecaltpp), shift_(0)
{
}

  EcalFenixStripFormatEB::~EcalFenixStripFormatEB() {
  }

  int EcalFenixStripFormatEB::setInput(int input, int inputPeak) 
  {
    inputPeak_=inputPeak;
    input_=input;
    return 0;
  }  
  
  int EcalFenixStripFormatEB::process()
  {
    buffer_=input_>>shift_;  //FIXME: buffer why?

    if(inputPeak_==0) return 0;
    int output=buffer_;
    if(output>0XFFF) output=0XFFF;   //ok: barrel saturates at 12 bits
    return output;    
  } 

//std::vector<int> EcalFenixStripFormatEB::process(std::vector<int> &peakout, std::vector<int> &filtout, std::vector<int> & output)
void EcalFenixStripFormatEB::process(std::vector<int> &peakout, std::vector<int> &filtout, std::vector<int> & output)
{
  //  std::vector<int> output;
  if  (peakout.size()!=filtout.size()){
    edm::LogWarning("EcalTPG")<<" problem in EcalFenixStripFormatEB: peak_out and filt_out don't have the same size";
  }
  for  (unsigned int i =0;i<filtout.size();i++){
    setInput(filtout[i],peakout[i]);
    //    int outone=process();
    //    output.push_back(outone);
    output[i]=process();
  }
  //  return output;
  return;
}

void EcalFenixStripFormatEB::setParameters(int SM, int towerInSM, int stripInTower)
{
  std::vector<unsigned int> const *params;
  params=ecaltpp_->getStripParameters(SM, towerInSM, stripInTower,params) ;
  shift_ = (*params)[0] ;
}
