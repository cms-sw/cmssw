#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CondFormats/L1TObjects/interface/EcalTPParameters.h"

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

//-------------------------------------------------------------------------------------
EcalFenixStrip::EcalFenixStrip(const EcalTPParameters *ecaltpp, const EcalElectronicsMapping* theMapping,bool debug, bool famos,int maxNrSamples) : theMapping_(theMapping), debug_(debug), famos_(famos)
{ 
  linearizer_.resize(EcalTPParameters::nbMaxXtals_);
  for (int i=0;i<EcalTPParameters::nbMaxXtals_;i++) linearizer_[i] = new  EcalFenixLinearizer(ecaltpp,famos_); 
  adder_ = new  EcalFenixEtStrip();
  amplitude_filter_ = new EcalFenixAmplitudeFilter(ecaltpp);
  peak_finder_ = new  EcalFenixPeakFinder();
  fenixFormatterEB_ = new EcalFenixStripFormatEB(ecaltpp);
  fenixFormatterEE_ = new EcalFenixStripFormatEE(ecaltpp);
  fgvbEE_ = new EcalFenixStripFgvbEE(ecaltpp);

  // prepare data storage for all events
  std::vector <int> v;
  v.resize(maxNrSamples);
  lin_out_.resize(EcalTPParameters::nbMaxXtals_);  
  for (int i=0;i<5;i++) lin_out_[i]=v;
  add_out_.resize(maxNrSamples);
  filt_out_.resize(maxNrSamples);
  peak_out_.resize(maxNrSamples);
  format_out_.resize(maxNrSamples);
  fgvb_out_.resize(maxNrSamples);
}

//-------------------------------------------------------------------------------------
EcalFenixStrip::~EcalFenixStrip() {
  for (int i=0;i<EcalTPParameters::nbMaxXtals_;i++) delete linearizer_[i]; 
  delete adder_; 
  delete amplitude_filter_; 
  delete peak_finder_;
  delete fenixFormatterEB_;
  delete fenixFormatterEE_;
  delete fgvbEE_;
}

//----------------------------------------------------------------------------------
void EcalFenixStrip::process_part2_barrel(int smnr,int townr,int stripnr) {
  
  // call formatter
  this->getFormatterEB()->setParameters(smnr, townr,stripnr) ; 
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
void  EcalFenixStrip::process_part2_endcap(int smnr,int townr,int stripnr) {
   
  // call  Fgvb
  this->getFGVB()->setParameters(smnr,townr,stripnr);
  this->getFGVB()->process(lin_out_,fgvb_out_);

  // call formatter
  this->getFormatterEE()->setParameters(smnr, townr,stripnr) ; 

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
//------------------------------------------------------------------------------------
int EcalFenixStrip::getCrystalNumberInStrip(const EBDataFrame *frame,int crystalPos)  {
  int crystalNumberInStrip=((frame->id()).ic()-1)%EcalTPParameters::nbMaxXtals_;
  if ((frame->id()).ieta()<0) crystalNumberInStrip=EcalTPParameters::nbMaxXtals_ - crystalNumberInStrip - 1;
  crystalNumberInStrip++;
  return crystalNumberInStrip;
}
//--------------------------------------------------------------------------------------

int EcalFenixStrip::getCrystalNumberInStrip(const EEDataFrame *frame,int crystalPos) {
  const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(frame->id());
  return elId.channelId();
}

//----------------------------------------------------------------------------------------

