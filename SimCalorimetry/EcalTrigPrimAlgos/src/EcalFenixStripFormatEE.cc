#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>
#include <CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>


//-----------------------------------------------------------------------------------------
EcalFenixStripFormatEE::EcalFenixStripFormatEE() 
  : shift_(0)
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
  //  buffer_=input_>>shift_;
    
  //  int output=buffer_;
  int output=input_>>shift_;
//barrel saturates at 12 bits, endcap at 10!
// Pascal: finally no,endcap has 12 bits as in EB (bug in FENIX!!!!)
  if(output>0XFFF) output=0XFFF; 
  output=output|(fgvb_<<12); //Pascal (was 10)

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

void EcalFenixStripFormatEE::setParameters(uint32_t id,const EcalTPGSlidingWindow*& slWin){

  const EcalTPGSlidingWindowMap &slwinmap = slWin -> getMap();
  EcalTPGSlidingWindowMapIterator it=slwinmap.find(id);
  if (it!=slwinmap.end()) shift_=(*it).second;
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGSlidingWindowMap entry for "<<id;
}
//-----------------------------------------------------------------------------------------
