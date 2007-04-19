#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormat.h>
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

  EcalFenixStripFormat::EcalFenixStripFormat(const EcalTPParameters *ecaltpp)
    : ecaltpp_(ecaltpp), shift_(0)
{
}

  EcalFenixStripFormat::~EcalFenixStripFormat() {
  }

  int EcalFenixStripFormat::setInput(int input, int inputPeak) 
  {
    inputPeak_=inputPeak;
    input_=input;
    return 0;
  }  
  
  int EcalFenixStripFormat::process()
  {

    buffer_[0]=input_>>shift_;

    if(inputPeak_==0) return 0;
    int output=buffer_[0];
    if(output>0XFFF) output=0XFFF;
    return output;    
  } 

std::vector<int> EcalFenixStripFormat::process(std::vector<int> peakout, std::vector<int> filtout)
{
  std::vector<int> output;
  if  (peakout.size()!=filtout.size()){
    edm::LogWarning("")<<" problem in EcalFenixStripFormat: peak_out and filt_out don't have the same size";
  }
  for  (unsigned int i =0;i<filtout.size();i++){
    setInput(filtout[i],peakout[i]);
    int outone=process();
    output.push_back(outone);
  }
  return output;
}

void EcalFenixStripFormat::setParameters(int SM, int towerInSM, int stripInTower)
{
  std::vector<unsigned int> params = ecaltpp_->getStripParameters(SM, towerInSM, stripInTower) ;
  shift_ = params[0] ;
}
