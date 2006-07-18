#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalBarrelFenixStrip.h>
#include <DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <TTree.h>

EcalBarrelFenixStrip::EcalBarrelFenixStrip(const TTree *tree) { 
  for (int i=0;i<nCrystalsPerStrip_;i++) linearizer_[i] = new  EcalFenixLinearizer(); 
  adder_ = new  EcalFenixEtStrip();
  amplitude_filter_ = new EcalFenixAmplitudeFilter();
  peak_finder_ = new  EcalFenixPeakFinder();
  formatter_= new EcalFenixStripFormat();
}


EcalBarrelFenixStrip::~EcalBarrelFenixStrip() {
  for (int i=0;i<nCrystalsPerStrip_;i++) delete linearizer_[i]; 
  delete adder_; 
  delete amplitude_filter_; 
  delete peak_finder_;
  delete formatter_;
}


std::vector<int>  EcalBarrelFenixStrip::process(std::vector<const EBDataFrame *> & df, int stripnr, int townr)
{ 

  // call linearizer
  std::vector<EBDataFrame * > lin_out;
  //this is a test:
  //    cout<<"EcalBarrelFenixStrip input is a vector of size: "<<df.size()<<endl;
  for (unsigned int cryst=0;cryst<df.size();cryst++) {
    //this is a test:
    //       cout<<endl;
    //       cout <<"cryst= "<<cryst<<" EBDataFrame is: "<<endl; 
    //       for ( int i = 0; i<cc.size();i++){
    // 	cout <<" "<<df[cryst][i];
    // 	if (df[cryst][i].adc() > 210) cout<<" is great!!";
    //       }
    //       cout<<endl;
    EBDataFrame *ebdfp= new EBDataFrame(df[cryst]->id());
    this->getLinearizer(cryst)->process(*(df[cryst]),stripnr,townr,ebdfp);
    lin_out.push_back(ebdfp);
  }

  //this is a test:
  //     cout<< "output of linearizers is a vector of size: "<<lin_out.size()<<endl; 
  //     for (unsigned int ix=0;ix<lin_out.size();ix++){
  //       cout<< "cryst: "<<ix<<"  value : "<<endl;
  //       for (int i =0; i<lin_out[ix].size();i++){
  // 	cout <<" "<<lin_out[ix][i];
  //       }
  //     }
		
  //    cout<<endl;
  // call adder
  std::vector<int> add_out;
  add_out = this->getAdder()->process(lin_out);
  for (unsigned int i=0;i<lin_out.size();++i) delete lin_out[i];

  //     //this is a test:
  //     cout<< "output of adder is a vector of size: "<<add_out.size()<<endl; 
  //     cout<< "value : "<<endl;
  //     for (unsigned int i =0; i<add_out.size();i++){
  //       cout <<" "<<add_out[i];
  //     }
      
  //     cout<<endl;
 
  // call amplitudefilter
  std::vector<int> filt_out;
  filt_out= this->getFilter()->process(add_out);
  //this is a test:
  //     cout<< "output of amplitude filter is a vector of size: "<<filt_out.size()<<endl; 
  //     cout<< "value : "<<endl;
  //     for (unsigned int i =0; i<filt_out.size();i++){
  //       cout <<" "<<filt_out[i];
  //     }    
  //     cout<<endl;

  // call peakfinder
  std::vector<int> peak_out;
  peak_out =this->getPeakFinder()->process(filt_out);

  //this is a test:
  //     cout<< "output of peak finder is a vector of size: "<<peak_out.size()<<endl; 
  //     cout<< "value : "<<endl;
  //     for (unsigned int i =0; i<peak_out.size();i++){
  //       cout <<" "<<peak_out[i];
  //     }    
  //     cout<<endl;

  // call formatter
  std::vector<int> format_out(peak_out.size());
  format_out =this->getFormatter()->process(peak_out,filt_out);
     
  //this is a test:
  //     cout<< "output of formatter is a vector of size: "<<format_out.size()<<endl; 
  //     cout<< "value : "<<endl;
  //     for (unsigned int i =0; i<format_out.size();i++){
  //       cout <<" "<<format_out[i];
  //     }    
  //    cout<<endl;

  return format_out;

}

