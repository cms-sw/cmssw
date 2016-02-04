#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

//-------------------------------------------------------------------------------------
EcalFenixStrip::EcalFenixStrip(const edm::EventSetup & setup, const EcalElectronicsMapping* theMapping,bool debug, bool famos,int maxNrSamples, int nbMaxXtals): theMapping_(theMapping), debug_(debug), famos_(famos), nbMaxXtals_(nbMaxXtals)
{ 
  linearizer_.resize(nbMaxXtals_);
  for (int i=0;i<nbMaxXtals_;i++) linearizer_[i] = new  EcalFenixLinearizer(famos_); 
  adder_ = new  EcalFenixEtStrip();
  amplitude_filter_ = new EcalFenixAmplitudeFilter();
  peak_finder_ = new  EcalFenixPeakFinder();
  fenixFormatterEB_ = new EcalFenixStripFormatEB();
  fenixFormatterEE_ = new EcalFenixStripFormatEE();
  fgvbEE_ = new EcalFenixStripFgvbEE();

  // prepare data storage for all events
  std::vector <int> v;
  v.resize(maxNrSamples);
  lin_out_.resize(nbMaxXtals_);  
  for (int i=0;i<5;i++) lin_out_[i]=v;
  add_out_.resize(maxNrSamples);
  filt_out_.resize(maxNrSamples);
  peak_out_.resize(maxNrSamples);
  format_out_.resize(maxNrSamples);
  fgvb_out_.resize(maxNrSamples);
}

//-------------------------------------------------------------------------------------
EcalFenixStrip::~EcalFenixStrip() {
  for (int i=0;i<nbMaxXtals_;i++) delete linearizer_[i]; 
  delete adder_; 
  delete amplitude_filter_; 
  delete peak_finder_;
  delete fenixFormatterEB_;
  delete fenixFormatterEE_;
  delete fgvbEE_;
}

//----------------------------------------------------------------------------------
void EcalFenixStrip::process_part2_barrel(uint32_t stripid,const EcalTPGSlidingWindow * ecaltpgSlidW) {
  
  // call formatter
  this->getFormatterEB()->setParameters(stripid,ecaltpgSlidW) ; 
  this->getFormatterEB()->process(peak_out_,filt_out_,format_out_);     
  //this is a test:
  if (debug_) {
    std::cout<< "output of formatter is a vector of size: "<<format_out_.size()<<std::endl; 
    std::cout<< "value : "<<std::endl;
    for (unsigned int i =0; i<format_out_.size();i++){
      std::cout <<" "<<format_out_[i];
    }    
    std::cout<<std::endl;

  }
  return;

}
//-------------------------------------------------------------------------------------
void  EcalFenixStrip::process_part2_endcap(uint32_t stripid,const EcalTPGSlidingWindow * ecaltpgSlidW,const EcalTPGFineGrainStripEE * ecaltpgFgStripEE) {
   
  // call  Fgvb
  this->getFGVB()->setParameters(stripid,ecaltpgFgStripEE); 
  this->getFGVB()->process(lin_out_,fgvb_out_);

  // call formatter
  this->getFormatterEE()->setParameters(stripid,ecaltpgSlidW) ;

  this->getFormatterEE()->process(fgvb_out_,peak_out_,filt_out_,format_out_);
     
  //this is a test:
   if (debug_) {
     std::cout<< "output of formatter is a vector of size: "<<format_out_.size()<<std::endl; 
      std::cout<< "value : "<<std::endl;
      for (unsigned int i =0; i<format_out_.size();i++){
        std::cout <<" "<<std::dec<<format_out_[i];
      }    
     std::cout<<std::endl;
   }

   return;
}
