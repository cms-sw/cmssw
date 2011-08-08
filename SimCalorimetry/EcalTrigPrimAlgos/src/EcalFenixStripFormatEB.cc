#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>
#include <CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

  EcalFenixStripFormatEB::EcalFenixStripFormatEB() 
    : shift_(0)
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
    //    buffer_=input_>>shift_;  //FIXME: buffer why?

    if(inputPeak_==0) return 0;
    //    int output=buffer_;
    int output=input_>>shift_;
    if(output>0XFFF) output=0XFFF;   //ok: barrel saturates at 12 bits
    return output;    
  } 

void EcalFenixStripFormatEB::process(std::vector<int> &peakout, std::vector<int> &filtout, std::vector<int> & output)
{
  if  (peakout.size()!=filtout.size()){
    edm::LogWarning("EcalTPG")<<" problem in EcalFenixStripFormatEB: peak_out and filt_out don't have the same size";
  }
  for  (unsigned int i =0;i<filtout.size();i++){
    setInput(filtout[i],peakout[i]);

    output[i]=process();
  }
  return;
}

void EcalFenixStripFormatEB::setParameters(uint32_t& id, const EcalTPGSlidingWindow*& slWin)
{

  const EcalTPGSlidingWindowMap &slwinmap = slWin -> getMap();
  EcalTPGSlidingWindowMapIterator it=slwinmap.find(id);
  if (it!=slwinmap.end()) shift_=(*it).second;
  else edm::LogWarning("EcalTPG")<<" could not find EcalTPGSlidingWindowMap entry for "<<id;
}
