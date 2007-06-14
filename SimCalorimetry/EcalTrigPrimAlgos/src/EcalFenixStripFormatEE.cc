#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>
#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


//-----------------------------------------------------------------------------------------
EcalFenixStripFormatEE::EcalFenixStripFormatEE(const EcalTPParameters * ecaltpp) 
  : ecaltpp_(ecaltpp), shift_(0)
{
}
//------------------------------------------------------------------------------------------

EcalFenixStripFormatEE::~EcalFenixStripFormatEE() {
}

//-----------------------------------------------------------------------------------------


int EcalFenixStripFormatEE::setInput(int input, int inputPeak, int fgvb ) {
  inputPeak_=inputPeak;
  input_=input;
  fgvb_=fgvb;
  return 0;
}  
//-----------------------------------------------------------------------------------------
  
int EcalFenixStripFormatEE::process(){
  if(inputPeak_==0) return 0;
  buffer_=input_>>shift_;
    
  int output=buffer_;
  //barrel saturates at 12 bits, endcap at 10!
  if(output>0X3FF) output=0X3FF; 
  output=output|(fgvb_<<10); //FIXME

  return output;    
} 
//------------------------------------------------------------------------------------------

void EcalFenixStripFormatEE::process(std::vector<int> &fgvbout,std::vector<int> &peakout,
				     std::vector<int> &filtout, std::vector<int> &output){
  if  (peakout.size()!=filtout.size()){
    edm::LogWarning("EcalTPG")<<" problem in EcalFenixStripFormatEE: peak_out and filt_out don't have the same size";
    std::cout<<" Size peak_out"<< peakout.size()<<", size filt_out:"<<filtout.size()<<std::flush<<std::endl;
  }
  for  (unsigned int i =0;i<filtout.size();i++){
    setInput(filtout[i],peakout[i],fgvbout[i]);
    output[i]=process();
  }
  return;
}
//-----------------------------------------------------------------------------------------

void EcalFenixStripFormatEE::setParameters(int sector, int towerInSector, int stripInTower){
  std::vector<unsigned int> const *params;
  params= ecaltpp_->getStripParameters(sector, towerInSector, stripInTower) ;
  shift_ = (*params)[0] ;
}
//-----------------------------------------------------------------------------------------
