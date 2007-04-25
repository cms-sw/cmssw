#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>

#include <TTree.h>
//-------------------------------------------------------------------------------------
EcalFenixStrip::EcalFenixStrip(TTree *tree, const EcalTPParameters *ecaltpp, const EcalElectronicsMapping* theMapping,bool debug) : theMapping_(theMapping), debug_(debug)
{ 
  for (int i=0;i<nCrystalsPerStrip_;i++) linearizer_[i] = new  EcalFenixLinearizer(ecaltpp); 
  adder_ = new  EcalFenixEtStrip();
  amplitude_filter_ = new EcalFenixAmplitudeFilter(ecaltpp);
  peak_finder_ = new  EcalFenixPeakFinder();
  fenixFormatterEB_ = new EcalFenixStripFormatEB(ecaltpp);
  fenixFormatterEE_ = new EcalFenixStripFormatEE(ecaltpp);
  fgvbEE_ = new EcalFenixStripFgvbEE(ecaltpp);
}

//-------------------------------------------------------------------------------------
EcalFenixStrip::~EcalFenixStrip() {
  for (int i=0;i<nCrystalsPerStrip_;i++) delete linearizer_[i]; 
  delete adder_; 
  delete amplitude_filter_; 
  delete peak_finder_;
  delete fenixFormatterEB_;
  delete fenixFormatterEE_;
  delete fgvbEE_;
}

//----------------------------------------------------------------------------------
std::vector<int>  EcalFenixStrip::process_part2_barrel(std::vector<int> &peak_out,std::vector<int> &filt_out,int smnr,int townr,int stripnr) {
  
  // call formatter
  this->getFormatterEB()->setParameters(smnr, townr,stripnr) ; // PP: sm number must be here instead of 1
  std::vector<int> format_out(peak_out.size());
  format_out =this->getFormatterEB()->process(peak_out,filt_out);
     
  //this is a test:
  //     cout<< "output of formatter is a vector of size: "<<format_out.size()<<endl; 
  //     cout<< "value : "<<endl;
  //     for (unsigned int i =0; i<format_out.size();i++){
  //       cout <<" "<<format_out[i];
  //     }    
  //    cout<<endl;

  std::cout<<"fin process_2   barrel ++++++"<<std::flush<<std::endl;
  return format_out;

}
//-------------------------------------------------------------------------------------
std::vector<int>  EcalFenixStrip::process_part2_endcap(std::vector<const EEDataFrame *> &lin_out,std::vector<int> &peak_out,std::vector<int> &filt_out,int smnr,int townr,int stripnr) {
  
  // call  Fgvb
  std::cout<<"dans EcalFenixStrip::process_part2_endcap"<<std::flush<<std::endl;
  std::vector<int> fgvb_out;
  this->getFGVB()->setParameters(smnr,townr,stripnr);
  fgvb_out = this->getFGVB()->process(lin_out);

  // call formatter
  this->getFormatterEE()->setParameters(smnr, townr,stripnr) ; // PP: sm number must be here instead of 1
  std::vector<int> format_out(peak_out.size());
  format_out =this->getFormatterEE()->process(fgvb_out,peak_out,filt_out);
     
  //this is a test:
   if (debug_) {
     std::cout<< "output of formatter is a vector of size: "<<format_out.size()<<std::endl; 
      std::cout<< "value : "<<std::endl;
      for (unsigned int i =0; i<format_out.size();i++){
        std::cout <<" "<<std::dec<<format_out[i];
      }    
     std::cout<<std::endl;
   }

  std::cout<<"fin process_2   endcap ++++++"<<std::flush<<std::endl;
  return format_out;

}
//------------------------------------------------------------------------------------
int EcalFenixStrip::getCrystalNumberInStrip(const EBDataFrame *frame,int crystalPos)  {
  int crystalNumberInStrip=((frame->id()).ic()-1)%numberOfCrystalsInStrip;
  if ((frame->id()).ieta()<0) crystalNumberInStrip=numberOfCrystalsInStrip - crystalNumberInStrip - 1;
  crystalNumberInStrip++;
  return crystalNumberInStrip;
}
//--------------------------------------------------------------------------------------

int EcalFenixStrip::getCrystalNumberInStrip(const EEDataFrame *frame,int crystalPos) {
  const EcalTriggerElectronicsId elId = theMapping_->getTriggerElectronicsId(frame->id());
  return elId.channelId();
}

//----------------------------------------------------------------------------------------
// template <class T> std::vector<int>  EcalFenixStrip::process(std::vector<const T *> & df, int stripnr, int townr)
// { 

//   // call linearizer
//   std::vector<T * > lin_out;
//   //this is a test:
//   //    cout<<"EcalFenixStrip input is a vector of size: "<<df.size()<<endl;
//   for (unsigned int cryst=0;cryst<df.size();cryst++) {
//     //this is a test:
//     //       cout<<endl;
//     //       cout <<"cryst= "<<cryst<<" EBDataFrame is: "<<endl; 
//     //       for ( int i = 0; i<cc.size();i++){
//     // 	cout <<" "<<df[cryst][i];
//     // 	if (df[cryst][i].adc() > 210) cout<<" is great!!";
//     //       }
//     //       cout<<endl;
//     int crystalNumberInStrip=((df[cryst]->id()).ic()-1)%numberOfCrystalsInStrip;
//     if ((df[cryst]->id()).ieta()<0) crystalNumberInStrip=numberOfCrystalsInStrip - crystalNumberInStrip - 1;
//     crystalNumberInStrip++;    // to start with 1 for DBInterface
//     this->getLinearizer(cryst)->setParameters(1, townr, stripnr, crystalNumberInStrip) ; // PP: sm number must be here instead of 1
//     //    EBDataFrame *ebdfp= new EBDataFrame(df[cryst]->id());
//     T *ebdfp= new T(df[cryst]->id());
//     this->getLinearizer(cryst)->process(*(df[cryst]),ebdfp);
//     lin_out.push_back(ebdfp);
//   }

//   //this is a test:
//   //     cout<< "output of linearizers is a vector of size: "<<lin_out.size()<<endl; 
//   //     for (unsigned int ix=0;ix<lin_out.size();ix++){
//   //       cout<< "cryst: "<<ix<<"  value : "<<endl;
//   //       for (int i =0; i<lin_out[ix].size();i++){
//   // 	cout <<" "<<lin_out[ix][i];
//   //       }
//   //     }
		
//   //    cout<<endl;
//   // call adder
//   std::vector<int> add_out;
//   add_out = this->getAdder()->process(lin_out);
//   for (unsigned int i=0;i<lin_out.size();++i) delete lin_out[i];

//   //     //this is a test:
//   //     cout<< "output of adder is a vector of size: "<<add_out.size()<<endl; 
//   //     cout<< "value : "<<endl;
//   //     for (unsigned int i =0; i<add_out.size();i++){
//   //       cout <<" "<<add_out[i];
//   //     }
      
//   //     cout<<endl;
 
//   // call amplitudefilter
//   this->getFilter()->setParameters(1, townr,stripnr) ; // PP: sm number must be here instead of 1
//   std::vector<int> filt_out;
//   filt_out= this->getFilter()->process(add_out); 
//   //this is a test:
//   //     cout<< "output of amplitude filter is a vector of size: "<<filt_out.size()<<endl; 
//   //     cout<< "value : "<<endl;
//   //     for (unsigned int i =0; i<filt_out.size();i++){
//   //       cout <<" "<<filt_out[i];
//   //     }    
//   //     cout<<endl;

//   // call peakfinder
//   std::vector<int> peak_out;
//   peak_out =this->getPeakFinder()->process(filt_out);

//   //this is a test:
//   //     cout<< "output of peak finder is a vector of size: "<<peak_out.size()<<endl; 
//   //     cout<< "value : "<<endl;
//   //     for (unsigned int i =0; i<peak_out.size();i++){
//   //       cout <<" "<<peak_out[i];
//   //     }    
//   //     cout<<endl;

//   // call formatter
//   this->getFormatter()->setParameters(1, townr,stripnr) ; // PP: sm number must be here instead of 1
//   std::vector<int> format_out(peak_out.size());
//   format_out =this->getFormatter()->process(peak_out,filt_out);
     
//   //this is a test:
//   //     cout<< "output of formatter is a vector of size: "<<format_out.size()<<endl; 
//   //     cout<< "value : "<<endl;
//   //     for (unsigned int i =0; i<format_out.size();i++){
//   //       cout <<" "<<format_out[i];
//   //     }    
//   //    cout<<endl;

//   return format_out;

// }

